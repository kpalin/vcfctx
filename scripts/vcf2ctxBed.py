#!/usr/bin/env python
"""Calculate the mutation spectrum for tumors, i.e. the number of mutations
of given type or Produce a bed file for mutations with mutation sequence context"""
try:
    import vcf
except ImportError as e:
    print("Please run: 'pip install pyvcf' or 'pip install --user pyvcf'")
    raise e


class AdjacentSubMergeReader(vcf.Reader):
    """Read vcf files but merge adjacent substitution sites to single
    multinucleotide substitution
    """

    def __init__(self, *args, **kwargs):
        super(AdjacentSubMergeReader, self).__init__(*args, **kwargs)
        self._Reader_next = super(AdjacentSubMergeReader, self).__next__
        self._prevRecord = None
        self._prevDone = False

    def _canMerge(self, this, that):
        """Return true if Record this is exactly adjacent to that and look like forming a haplotype.

        Arguments:
        - `this`:
        - `that`:
        """
        import logging as log

        posOK = this.CHROM == that.CHROM and (this.POS + len(this.REF)) == that.POS
        if not posOK:
            return False

        if len(this.ALT) != len(that.ALT):
            log.warning(
                "Alternative allele count mismatch: %s %s " % (str(this), str(that))
            )
            return False

        from vcf.model import _Substitution

        if not all(isinstance(X, _Substitution) for X in this.ALT + that.ALT):
            log.warning(
                "Cant merge sites %s and %s. Can only merge substitutions."
                % (str(this), str(that))
            )
            return False

        if this.samples is None or that.samples is None:
            return False

        for iS, aS in zip(this.samples, that.samples):
            if iS.gt_type != aS.gt_type:
                log.info(
                    "GT mismatch on %s and %s for sample %s: %s<->%s"
                    % (str(this), str(that), iS.sample, iS.gt_bases, aS.gt_bases)
                )
                return False
        return True

    def _doMerge(self, this, that):
        """Merge sites this and that (assumes that _canMerge())

        Arguments:
        - `this`:
        - `that`:
        """
        assert len(this.alleles) == 2
        assert len(that.alleles) == 2
        from vcf.model import _Substitution

        newALT = _Substitution(this.ALT[0].sequence + that.ALT[0].sequence)
        newREF = this.REF + that.REF
        import copy
        import logging as log

        newRecord = copy.deepcopy(this)
        newRecord.REF = newREF
        newRecord.ALT = [newALT]
        newRecord.alleles = [newREF, newALT]

        log.info("Merged %s and %s to %s" % (str(this), str(that), str(newRecord)))
        return newRecord

    def __next__(self):
        """Return next (substitution merged) record from file"""
        mged = self._prevRecord
        while True:
            try:
                r = self._Reader_next()
            except StopIteration:  # Handle the end of the file
                if self._prevRecord is None:
                    raise
                else:
                    self._prevRecord = None
                    break

            self._prevRecord = r

            if mged is None:
                mged = self._prevRecord
            else:
                if self._canMerge(mged, r):
                    mged = self._doMerge(mged, r)
                else:
                    break

        return mged


class SiteSequenceContext(object):
    """Scan through a vcf file and look for context of the mutations"""

    def __init__(self, vcfFile, refFile, mergeAdjacent=True):
        """

        Arguments:
        - `vcfFile`: Filename or a stream for VCF
        - `refFile`: Filename for reference sequence in fasta format
        - `mergeAdjacent`: Merge 'double' substitutions in to one.
        """
        self._vcfFile = vcfFile
        self._refFile = refFile

        if mergeAdjacent:
            Reader = AdjacentSubMergeReader
        else:
            from vcf import Reader

        if isinstance(vcfFile, str):
            self.vcf = Reader(filename=self._vcfFile, strict_whitespace=True)
        else:
            self.vcf = Reader(self._vcfFile, strict_whitespace=True)

        import hugeFasta

        self.genome = hugeFasta.hugeFasta(self._refFile)

        self.tumors = dict((sample, None) for sample in self.vcf.samples)

    def iterContextMutations(self, context=1, contig=None):
        """Iterate over mutations with context information. Returns (site, {sample:(fromCtx,toCtx)])

        Arguments:
        - `contig`:
        """

        if contig is not None:
            self.vcf = self.vcf.fetch(contig, 0, 1000000000)

        from logging import info, warning
        from collections import defaultdict
        from vcf.model import _Breakend, _Substitution

        latestCall = defaultdict(
            int
        )  # Position of the previous call for the tumor sample
        warnedMissingChroms = set()
        for site in self.vcf:
            spectra = dict((t, list()) for t in self.tumors.keys())
            chrom, pos, refA = site.CHROM, site.POS - 1, site.REF
            try:
                refSeq = self.genome[chrom]
            except KeyError:
                if chrom not in warnedMissingChroms:
                    warning("Can't find reference chromosome '%s'" % (chrom))
                    warnedMissingChroms.add(chrom)
                continue
            refCtx = refSeq[(pos - context) : (pos + len(refA) + context)]
            refCtx = refCtx.upper()

            prime5 = refCtx[:context]
            prime3 = refCtx[-context:]

            if self.tumors.get("REF", None) == "ALT":
                altAs = set(
                    a.sequence for a in site.alleles[1:] if isinstance(a, _Substitution)
                )
                spectra["REF"] = []
                if len(altAs) > 1:
                    info(
                        "A lot of alternative alleles on %s:%d  %s %s"
                        % (site.CHROM, site.POS, refCtx, str(altAs))
                    )
                for k in ((refCtx, prime5 + A + prime3) for A in altAs):
                    spectra["REF"].append(k)

            else:
                for tumor, normal in self.tumors.items():
                    sample = site.genotype(tumor)
                    if sample.called:
                        sampleGeno = set(sample.gt_alleles)
                        filterGeno = set("0")
                        if normal is not None:
                            normalSample = site.genotype(normal)
                            if normalSample.called:
                                filterGeno = set(normalSample.gt_alleles)

                        altAs = set(
                            site.alleles[int(X)].sequence
                            for X in sampleGeno - filterGeno
                            if isinstance(site.alleles[int(X)], _Substitution)
                        )

                        if len(filterGeno) == 1:  # Hom ref -> het mut
                            refAidx = int(list(filterGeno)[0])
                        else:  # het ref -> hom mut
                            refAidx = list(filterGeno - sampleGeno)
                            if len(refAidx) == 0:
                                info(
                                    "Weird thing: Filtering %s with %s"
                                    % (str(filterGeno), str(sampleGeno))
                                )
                                continue
                            refAidx = int(refAidx[0])

                        refA = str(site.alleles[refAidx])

                        if refCtx[context:-(context)] != refA or refAidx != 0:
                            info(
                                "Something funny at %s:%d  %s %s refIdx:%d "
                                % (site.CHROM, site.POS, refCtx, str(refA), refAidx)
                            )

                        wtCtx = prime5 + refA + prime3
                        if len(altAs) == 0:
                            info("Ignoring site %s" % (str(site)))
                            continue
                        elif len(altAs) == 1:
                            if latestCall[tumor] >= pos:
                                warning(
                                    "Tumor %s has calls on adjacent or overlapping sites around %s:%d"
                                    % (tumor, chrom, pos + 1)
                                )
                        elif len(altAs) > 1:
                            info(
                                "Many mutations for sample %s on site %s"
                                % (str(sample), str(site))
                            )
                        latestCall[tumor] = pos + max(len(A) for A in altAs)
                        for k in ((wtCtx, prime5 + A + prime3) for A in altAs):
                            spectra[tumor].append(k)
            yield (site, spectra)

    def countSpectrum(self, context=1, contig=None):
        """Calculate mutation spectrum for all tumors, possibly restricting to given contig

        Arguments:
        - `contig`:
        """

        from mylib import revComplement
        from collections import defaultdict

        # self.spectra=dict((t,defaultdict(int)) for t in self.tumors.iterkeys())
        self.spectra = dict()

        for site, spec in self.iterContextMutations(context, contig):
            for tumor, mutCtxes in spec.items():
                if tumor not in self.spectra:
                    self.spectra[tumor] = defaultdict(int)
                for k in mutCtxes:
                    revK0 = revComplement(k[0])
                    if revK0 < k[0]:
                        k = (revK0, revComplement(k[1]))
                    self.spectra[tumor][k] += 1

    def writeMutCtxBed(self, outstrm, context=1, contig=None):
        """Write bed file of mutations"""

        tumors = list(sorted(self.tumors.keys()))
        outstrm.write("#chrom\tpos\tpos1\t" + "\t".join(tumors) + "\n")

        for site, spec in self.iterContextMutations(context, contig):
            ctxMuts = "\t".join(
                ",".join(map(":".join, spec[tumor])) for tumor in tumors
            )
            outstrm.write(
                "%s\t%d\t%d\t%s\n" % (site.CHROM, site.POS - 1, site.POS, ctxMuts)
            )

    def write(self, outStrm):
        """

        Arguments:
        - `outStrm`:
        """
        mTypes = set()
        for s in self.spectra.values():
            mTypes.update(list(s.keys()))
        mTypes = list(sorted(mTypes))
        samples = list(sorted(self.tumors.keys()))
        outStrm.write("#mutType\t" + "\t".join(samples) + "\n")
        for mType in mTypes:
            outStrm.write(
                "%s>%s\t" % (mType)
                + "\t".join(str(self.spectra[sample][mType]) for sample in samples)
                + "\n"
            )
        outStrm.flush()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate mutation spectrum for given tumor normal pairs."
    )

    parser.add_argument(
        "-v", "--vcf", help="Input VCF file [default:%(default)s]", default="-"
    )
    parser.add_argument(
        "-r",
        "--reference",
        help="Reference fasta file [default:%(default)s]",
        default="/data/public/reference-genomes/hs37d5/hs37d5.fa",
    )

    parser.add_argument(
        "-p",
        "--pairs",
        help="File tumor:normal pairs on each line. If not given, reference is considered 'normal' and alternative alleles 'tumor' [default:%(default)s]",
    )

    parser.add_argument(
        "-c",
        "--context",
        type=int,
        default=1,
        help="Length of the sequence context on both sides of the variant [default:%(default)s]",
    )
    parser.add_argument(
        "-C", "--contig", help="Contig to process [default:%(default)s]"
    )
    parser.add_argument(
        "-V",
        "--verbose",
        default=False,
        action="store_true",
        help="Be more verbose with output [default:%(default)s]",
    )
    parser.add_argument(
        "-M",
        "--adjacentmerge",
        default=False,
        action="store_true",
        help="Merge adjacent 'double' mutations in to one if they always occur together [default:%(default)s]",
    )

    parser.add_argument(
        "-B",
        "--bedout",
        default=False,
        action="store_true",
        help="Instead of mutation spectrum, output a bed file giving the mutations and contexts [default:%(default)s]",
    )

    parser.add_argument(
        "-o",
        "--out",
        help="Output file for mutation spectra of all samples. One row for tumor sample and one column for one mutation type [default:%(default)s]",
        default="mutSpectrum.tsv",
    )

    args = parser.parse_args()

    import logging

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    seqContext = SiteSequenceContext(args.vcf, args.reference, args.adjacentmerge)

    if args.pairs is None:
        seqContext.tumors = {"REF": "ALT"}
    else:
        new_pairs = [x.strip().split(":") for x in open(args.pairs)]
        seqContext.tumors = {t: n for (t, n) in new_pairs if t in seqContext.tumors}
        logging.info(
            "Using %d tumor normal pairs: %s "
            % (len(seqContext.tumors), str(seqContext.tumors))
        )

    if args.out in ("-", "stdout"):
        import sys

        outStrm = sys.stdout
    else:
        outStrm = open(args.out, "w")

    if args.bedout:
        seqContext.writeMutCtxBed(outStrm, args.context, args.contig)
    else:
        seqContext.countSpectrum(args.context, args.contig)
        seqContext.write(outStrm)

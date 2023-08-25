"""Tools for using samtools faidx indexed fasta files in python"""

import mmap, types
import os
import six


class hugeFasta:
    """Class for working with huge (genome size) fasta files.

    Initialize with the name of the indexed fasta file.
    Fetch sequences like

    seq = fidxed["X"][100:1000]

    Remember: The first position of the sequence is 0. (Not the typical 1 in e.g. VCF files)"""

    if six.PY3:
        transTable = bytes(range(256))
    else:
        transTable = "".join(map(chr, range(256)))

    class idxedFasta:
        def __init__(self, fastaMap, name, seqLen, startByte, lineLenBP, lineLenByte):
            (
                self.name,
                self.seqLenBP,
                self.startByte,
                self.lineLenBP,
                self.lineLenByte,
            ) = (name, seqLen, startByte, lineLenBP, lineLenByte)
            self.fastaMap = fastaMap

            self.is_mmap = not hasattr(self.fastaMap, "seek")

            self.seqLenBP += 1
            self.lastByte = self.updateIndecies(self.seqLenBP - 1)

            # Works just in UNIX!!!!
            # self.fastaMap=mmap.mmap(fileno,self.seqLenBP*(self.lineLenByte-self.lineLenBP),mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE,self.startByte)

        def countExtraBytes(self, i):
            extraBytes = i / self.lineLenBP
            extraBytes *= self.lineLenByte - self.lineLenBP

            return extraBytes

        def updateIndecies(self, i):
            "Update indecies to take of line feeds"
            if isinstance(i, int):
                if i >= self.seqLenBP:
                    raise IndexError()
                i += self.countExtraBytes(i) + self.startByte

            elif isinstance(i, slice):
                nStart, nStop = i.start, i.stop
                if nStart is not None:
                    if nStart >= self.seqLenBP:
                        raise IndexError()
                    nStart += self.countExtraBytes(nStart) + self.startByte
                if nStop is not None:
                    if nStop >= self.seqLenBP:
                        raise IndexError()
                    nStop += self.countExtraBytes(nStop) + self.startByte
                i = slice(int(nStart), int(nStop))
            else:
                raise TypeError("Index should be either int or slice")

            return i

        def __len__(self):
            return self.seqLenBP

        def __getitem__(self, i):
            "Return the given position in the sequence"
            i = self.updateIndecies(i)
            if self.is_mmap:
                if isinstance(i, slice):
                    return self.fastaMap[i.start : i.stop].translate(
                        hugeFasta.transTable, b"\n\t \r"
                    )
                else:
                    return self.fastaMap[i]
            else:
                if isinstance(i, slice):
                    self.fastaMap.seek(i.start)
                    return self.fastaMap.read(i.stop - i.start).translate(
                        hugeFasta.transTable, b"\n\t \r"
                    )
                else:
                    self.fastaMap.seek(i)
                    return self.fastaMap.read(1).translate(
                        hugeFasta.transTable, b"\n\t \r"
                    )

        def __setitem__(self, i, s):
            "Resets the given position in the sequence"
            # Need to correct for line feeds in the end of the line
            if self.is_mmap:
                if isinstance(i, slice):
                    if isinstance(s, slice) and len(s) == i.stop - i.start:
                        pass
                    else:
                        raise TypeError(
                            "Changing sequence length is not supported! (yet?)"
                        )
                else:
                    self.fastaMap[self.updateIndecies(i)] = s
            else:
                raise IOError(
                    "Fasta file is opened for reading only. You need more Virtual Memory to allow random point edits"
                )

    def __init__(self, fastaName):
        self.fastaName = fastaName
        try:
            self.fasta = open(self.fastaName, "r+")
            self.fastaMap = mmap.mmap(self.fasta.fileno(), 0)
        except IOError:
            self.fasta = open(self.fastaName, "r")
            self.fastaMap = mmap.mmap(
                self.fasta.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ
            )
        except mmap.error as e:
            if e.errno == 12:
                self.fasta = open(self.fastaName, "r")
                self.fastaMap = self.fasta
                # self.fastaMap = None
            else:
                raise

        self.readFastaIndex()

    def readFastaIndex(self):
        self.faidxName = "%s.fai" % (self.fastaName)

        faidx = open(self.faidxName)
        self.fai = {}
        for line in faidx:
            sName, sLenBP, startPos, bpLineLen, byteLineLen = line.strip().split()
            self.fai[sName] = hugeFasta.idxedFasta(
                self.fastaMap,
                sName,
                int(sLenBP),
                int(startPos),
                int(bpLineLen),
                int(byteLineLen),
            )

        faidx.close()

    def __len__(self):
        "Return the number of sequences in the fasta index"
        return len(self.fai)

    def close(self):
        self.fasta.close()

    def __getitem__(self, x):
        r = self.fai[x]
        return r

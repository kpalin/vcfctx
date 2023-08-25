#!/usr/bin/env python
#-*- coding: utf-8 -*-
""" 

Created at Fri Feb  3 10:52:50 2017 by Kimmo Palin <kpalin@merit.ltdk.helsinki.fi>
"""


def subSumContext(spectrum, context):
    """Remove context from the mutType:s and sum the rows with identical types (without context)

    Arguments:
    - `spectrum`:
    - `context`:
    """
    import numpy as np

    c = context
    mutParts = (str(s).split(">") for s in spectrum["mutType"])
    clearedCtx = np.fromiter(
        ("%s>%s" % (f[c:-c], t[c:-c]) for f, t in mutParts), dtype="|S20")
    subContext = set()
    from mylib import revComplement

    for ctx in list(sorted(set(clearedCtx))):
        rCtx = ">".join(revComplement(x) for x in ctx.split(">"))
        if ctx < rCtx:
            subContext.add(ctx)
        else:
            subContext.add(rCtx)

    V = spectrum[list(spectrum.dtype.names[1:])].view(int)
    V.shape = (len(spectrum), -1)

    outSpec = np.zeros(len(subContext), dtype=spectrum.dtype)
    for i, ctx in enumerate(sorted(subContext)):
        rCtx = ">".join(revComplement(x) for x in ctx.split(">"))
        ctxRows = np.nonzero((clearedCtx == ctx) | (clearedCtx == rCtx))[0]
        outSpec[i] = (ctx, ) + tuple(V[ctxRows, ].sum(axis=0))
    return outSpec


'''
This is a recipe from
http://www.scipy.org/Cookbook/Matplotlib/Interactive_Plotting
'''

import math


class AnnoteFinder:
    """
    callback for matplotlib to display an annotation when points are clicked on.  The
    point which is closest to the click and within xtol and ytol is identified.

    Register this function like this:

    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)
    """

    def __init__(self, xdata, ydata, annotes, axis=None, xtol=None, ytol=None):
        import pylab
        import matplotlib

        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata)) / float(len(xdata))) / 2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata)) / float(len(ydata))) / 2
        self.xtol = xtol
        self.ytol = ytol
        if axis is None:
            self.axis = pylab.gca()
        else:
            self.axis = axis
        self.drawnAnnotations = {}
        self.links = []

    def distance(self, x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return (math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def __call__(self, event):
        log.info(str(event))
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
        if (self.axis is None) or (self.axis == event.inaxes):
            annotes = []
            for x, y, a in self.data:
                if (clickX - self.xtol < x < clickX + self.xtol) and (
                        clickY - self.ytol < y < clickY + self.ytol):
                    annotes.append(
                        (self.distance(x, clickX, y, clickY), x, y, a))
            if annotes:
                annotes.sort()
                distance, x, y, annote = annotes[0]
                self.drawAnnote(event.inaxes, x, y, annote)
                for l in self.links:
                    l.drawSpecificAnnote(annote)

    def drawAnnote(self, axis, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
        else:
            t = axis.text(
                x,
                y,
                " - %s" % (annote), )
            m = axis.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
        self.axis.figure.canvas.draw()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.axis, x, y, a)


class Signature(object):
    """Mutational signatures
    """

    def __init__(self, **kwargs):
        """
        """
        self.__dict__.update(kwargs)

    def transform(self, X):
        """Transform the data X according to the fitted model

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be transformed by the model

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """
        assert (not hasattr(self, "decompType")) or "NMF" in self.decompType

        from scipy.optimize import nnls
        W = np.zeros((X.shape[0], self.components_.shape[0]))
        for j in range(0, X.shape[0]):
            W[j, :], _ = nnls(self.components_.T, X[j, :])
        return W

    @classmethod
    def load(cls, pickleFile):
        """Load list of signatures from a pickle file

        Arguments:
        - `pickleFile`:
        """
        import pickle
        inList = pickle.load(
            pickleFile if hasattr(pickleFile, "read") else open(pickleFile,"rb"))
        outList = [cls(**x) for x in inList]
        return outList


    def summary(self):
        signatures, features = self.components_.shape

        r = {"signatures":signatures,
            "features":features,
            "mean_silhouette":self.totalSilhouette,
            "min_silhouette":min(self.meanComponentSilhouettes), 
            "frobenius_err":self.reconstruction_err_}
        return r

    def __str__(self):
        signatures, features = self.components_.shape
        return "Signature:signatures:%d:features:%d:mean:%g:min silhouette:%g: Frobenius_err:%g" % (
            signatures, features, self.totalSilhouette,
            min(self.meanComponentSilhouettes), self.reconstruction_err_)

        return "Signature(%d signatures, %d features, %g mean %g min silhouette, %g Frobenius err)" % (
            signatures, features, self.totalSilhouette,
            min(self.meanComponentSilhouettes), self.reconstruction_err_)


class Spectrum(object):
    """Class for somatic mutation spectra
    """

    def __init__(self, spectrum, context=1):
        """

        Arguments:
        - `spectrum`:
        """
        self.context = context
        self.spectrum = spectrum.copy()
        import numpy as np
        self.rank = 3
        self.decomp = None
        from sklearn import decomposition
        self.decompFun = None

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, spect):
        """Set spectrum and few views to it
        """
        from numpy.lib import recfunctions as rfn
        #self._spectrum=spect[np.argsort(spect["mutType"])]
        self._spectrum = spect.copy()
        self.sampleNames = np.array(list(spect.dtype.names[1:]))
        import re
        pat = re.compile("_(SNV|Indel)s_(TUMOR|NORMAL)$")
        self.labels = np.array([pat.sub("", x) for x in self.sampleNames])

        self._spectrum = self._spectrum
        self.V = self._spectrum[list(self.sampleNames)].copy()
        self.V = rfn.repack_fields(self.V).view(int)


        self.V.shape = (len(self._spectrum), -1)
        self.features, self.samples = self.V.shape
        self.Ns = self.V.sum(axis=0)
        self.colors = np.array(
            [
                x.split(", ")
                for x in
                "141, 211, 199; 255, 255, 179; 190, 186, 218; 251, 128, 114; 128, 177, 211; 253, 180, 98; 179, 222, 105; 252, 205, 229; 217, 217, 217; 188, 128, 189; 204, 235, 197; 255, 237, 111".
                split(";")
            ],
            dtype=float) / 255.0
        #        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        #['red','blue','green','yellow','magenta','purple']

        self.featPermutation = np.argsort(self.spectrum["mutType"])
        #self.featMutations = np.array([x[self.context:self.context+1]+">"+x[-self.context-1:-self.context] for x in self.spectrum["mutType"]])
        self.featMutations = np.array([
            (x[self.context:(-self.context)] + ">" +
            y[(self.context):(-self.context)])
            for x, y in (z.split(">") for z in self.spectrum["mutType"])
        ])
        from itertools import cycle
        self.mutationColors = dict(
            list(zip(sorted(set(self.featMutations)), cycle(self.colors))))
        self.featColors = np.array(
            [self.mutationColors[x] for x in self.featMutations])

        log.info("Using %d features and %d samples." %
                 (self.features, self.samples))

    def shuffle(self):
        "Shuffle the mutation counts"
        import numpy as np
        import logging as log
        
        log.info("Shuffling the mutation spectrum! Output is random!!!")
        Vorig = self.V.copy()
        spec = self.spectrum.copy()
        
        shuffle_names = self.sampleNames.copy()
        for m in range(len(self.spectrum)):
            np.random.shuffle(shuffle_names)
            for i,oi in zip(shuffle_names,self.sampleNames):
                spec[m][i] = self.spectrum[m][oi]
            #np.random.shuffle(spec[i,:])
            
        log.debug(str(spec))
        self.spectrum = spec
        log.info("Froebinus error added to input due to shuffle: {}".format(np.linalg.norm(Vorig-self.V)))
        log.debug(np.array_str(self.V))


    def setSignatures(self, sig):
        """Set give Signature for this spectrum

        Arguments:
        - `sig`: Signature() object
        """
        sigRank, sigFeatures = sig.components_.shape
        if sigFeatures > self.features:
            raise ValueError(
                "Signature has %d features where as spectrum only has %d" %
                (sigFeatures, self.features))
        elif sigFeatures < self.features:
            if (np.in1d(sig.mutTypes, self.spectrum["mutType"])).all():
                newSpec = np.zeros(
                    len(sig.mutTypes), dtype=self.spectrum.dtype)
                for i, mt in enumerate(sig.mutTypes):
                    newSpec[i] = self.spectrum[self.spectrum["mutType"] == mt]
                self.spectrum = newSpec
                self.rank = sigRank
            else:
                raise ValueError("Signature and spectrum features don't match")

        self.decompType = sig.decompType if hasattr(sig,
                                                    "decompType") else "NMF"
        self.transformed = sig.transform(self.V.T)
        self.decomp = sig

    def filterSpectrum(self, propSitesRemove=0.01):
        """Remove features together accounting for at most propSitesRemove

        Arguments:
        - `propSitesRemove`:
        """
        totPerFeat = self.V.sum(axis=1)
        countRemove = totPerFeat.sum() * propSitesRemove
        featsRemove = (
            np.array(sorted(totPerFeat)).cumsum() < countRemove).sum()
        specIdx = totPerFeat.argsort()[featsRemove:]
        newSpec = self._spectrum[specIdx]
        self.spectrum = newSpec

    def bootstrapV(self, bootIndividuals=False):
        """Return a bootstrapped sample from spectrum matrix such that
        total number of mutations stays the same but relative proportions
        vary a bit (Sample from multinomial)

        If bootIndividuals is True, then bootstrap also selection of input individuals.
        """
        from numpy.random import multinomial, choice
        import numpy as np
        V = self.V
        if bootIndividuals:
            Nsamples = V.shape[1]
            V = V[:, choice(Nsamples, Nsamples, replace=True)]
        Ns = V.sum(axis=0)
        props = (V * 1.0 / Ns)
        assert np.allclose(props.sum(axis=0), 1.0)

        outV = V.copy()
        outV[:] = 0.0
        for i, N in enumerate(Ns):
            outV[:, i] = multinomial(N, props[:, i])

        return outV

    def NMF(self, rank=3, sparseness=None):
        """Non-negative matrix factorization object for spectrum
        """
        from sklearn import decomposition
        self.rank = rank
        self.decomp = decomposition.NMF(
            n_components=self.rank,
            max_iter=2000, ).fit(self.V.T)
        self.transformed = self.decomp.transform(self.V.T)
        self.decompType = "NMF"
        return self.decomp

    def PCA(self, rank=3):
        """Principal components analysis for spectrum
        """
        from sklearn import decomposition, preprocessing
        self.rank = rank
        V_scaled = preprocessing.scale(np.array(self.V.T, dtype=float))

        self.decomp = decomposition.PCA(n_components=self.rank).fit(V_scaled)
        self.transformed = self.decomp.transform(V_scaled)
        self.decompType = "PCA"
        return self.decomp

    def ICA(self, rank=3, scale=True, V=None):
        """Independent components analysis for spectrum
        """
        from sklearn import decomposition, preprocessing
        self.rank = rank
        if V is None:
            V_scaled = self.V.T
        else:
            V_scaled = V
        if scale:
            V_scaled = preprocessing.scale(np.array(self.V.T, dtype=float))

        self.decomp = decomposition.FastICA(
            n_components=self.rank, max_iter=2000).fit(V_scaled)
        self.transformed = self.decomp.transform(V_scaled)
        self.decompType = "ICA"
        return self.decomp

    def findSignature(self, rank=4, maxBoots=400, bootIndividuals=False):
        """Find bootstrapped signature of given rank

        Arguments:
        - `rank`:
        """
        from sklearn import decomposition
        from sklearn.metrics import silhouette_samples
        from scipy.spatial import distance
        from scipy.optimize import linear_sum_assignment

        import logging as log
        assert 0 < rank < min(
            self.samples,
            self.features), "Rank must be positive and less than full."

        bootSamples = [list() for _ in range(rank)]
        bootSamples = np.zeros((maxBoots, rank, self.features))

        fullModel = self.decompFun(rank).fit(self.V.T)
        log.debug(np.array_str(self.V))
        
        myComponents = fullModel.components_
        assert myComponents.var(axis=1).min()>0, "Trying to extract more signatures than there could possibly be."
        for i in range(maxBoots):
            if i % 10 == 0:
                log.info("Fit %d" % (i + 1))
            bootV = self.bootstrapV(bootIndividuals)
            nmfV = self.decompFun(rank).fit(bootV.T)
            bootComponents = nmfV.components_
            dists = distance.cdist(myComponents, bootComponents, "cosine")
            assert np.isfinite(dists).all()
            # Not exactly as in Alexandrov et.al. define the clusters according to original components.
            _, assign = linear_sum_assignment(dists)
            bootSamples[i] = bootComponents[assign]
            myComponents = bootSamples[:i + 1].mean(
                axis=0)  # myComponents follow the sample mean
        r = Signature()

        r.mutTypes = self._spectrum["mutType"]
        r.components_ = myComponents
        r.bootSamples = bootSamples
        flatterView = np.reshape(bootSamples, ((maxBoots) * rank,
                                               self.features))
        flatterLabels = np.tile(np.arange(rank, dtype=int), maxBoots)
        silh = silhouette_samples(
            flatterView, flatterLabels, metric="cosine").reshape(
                (rank, maxBoots))
        r.totalSilhouette = silh.mean()
        r.meanComponentSilhouettes = silh.mean(axis=1)
        r.decompType = self.decompType

        exposure = r.transform(self.V.T)
        # Reconstruction error from the mean estimate
        r.reconstruction_err_ = np.linalg.norm(self.V - np.dot(r.components_.T,
                                                               exposure.T))

        self.bootedSignatures = r

        import logging as log
        log.info(str(self.decomp))
        return r

    def findAllSignatures(self,
                          maxRank=13,
                          reductionProp=0.01,
                          maxBoots=400,
                          outFname="bootstrapSignatures.pickle",
                          bootIndividuals=False):
        """Run most of the steps desribed in Alexandrov et.al.
        """
        self.filterSpectrum(reductionProp)

        signs = []
        import logging as log
        self.signatures = []
        import pickle
        import os
        import json

        for rank in range(2, maxRank + 1):
            log.info("Finding signatures rank %d" % (rank))
            sign = self.findSignature(rank, maxBoots, bootIndividuals)

            print(sign)
            self.signatures.append(sign)

            pickle.dump([x.__dict__ for x in self.signatures],
                        open(outFname + ".new", "wb"))

            

            inList = pickle.load(open(outFname + ".new", "rb"))

            errs = np.array([((x["components_"] - s.components_)
                              **2).sum().sum()
                             for x, s in zip(inList, self.signatures)])
            assert (
                errs < 0.001
            ).all(), "There's something fishy with pickling of the components."
            errs = np.array([(x["mutTypes"] == s.mutTypes).all()
                             for x, s in zip(inList, self.signatures)])
            assert errs.all(), "There is something fishy with mutation types."

            os.rename(outFname + ".new", outFname)

        json.dump([s.summary() for s in self.signatures],open(outFname+".json","w"),indent=3)

    def plotPie(self, i, title=True, ax=None, **kwargs):
        """Make a pie plot of mutations of sample i

        Arguments:
        - `i`:
        - `ax`:
        """
        import pylab
        if ax is None:
            ax = pylab.gca()

        if not isinstance(i, str):
            i = self.sampleNames[i]

        data = self.spectrum[i]

        piePatches, _ = ax.pie(data, colors=self.featColors, **kwargs)
        for p in piePatches:
            p.set_edgecolor("none")

        if title:
            label = self.labels[np.where(self.sampleNames == i)[0][0]]
            ax.set_title(label)
        return ax

    def plotCounts(self, ax=None):
        """Plot pure mutation counts per sample

        Arguments:
        - `ax`:
        """
        import pylab
        if ax is None:
            ax = pylab.gca()
        ax.bar(list(range(self.samples)), sorted(self.Ns))
        ax.set_title("Number of mutations")
        ax.set_ylabel("Count")
        ax.set_xlabel("Sample")
        ax.set_xticks(np.arange(self.samples))
        ax.set_xticklabels(
            self.labels, [np.argsort(self.Ns)], rotation="vertical")
        return ax

    def plotSpectrum(self, ax=None):
        """Plot the counts of individual mutation types for all samples

        Arguments:
        - `spectrum`:
        - `ax`:
        """
        import pylab
        if ax is None:
            ax = pylab.gca()

        mutLabels = self.spectrum["mutType"]
        Ndim, Nsamples = self.features, self.samples

        Y = np.repeat(
            [np.arange(Ndim)], Nsamples, axis=0).T + (np.random.random(
                (Ndim, Nsamples)) - 0.5) * 0.3
        ax.scatter(Y, self.V)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-1, Ndim + 1)
        ax.set_ylabel("Count")
        ax.set_title("Mutation counts by type")
        ax.set_xticks(np.arange(Ndim))
        ax.set_xticklabels(mutLabels, rotation="vertical")
        Ytol = max(ax.get_ylim()) / 100.0

        af = AnnoteFinder(
            np.reshape(Y, (-1, )),
            np.reshape(self.V, (-1, )),
            np.tile(self.labels, self.features),
            ytol=Ytol,
            xtol=0.25,
            axis=ax)
        pylab.connect('button_press_event', af)
        return ax

    def plotSpectrumSample(self, sample, ax=None):
        """Plot sample spectrum with context matrix

        Arguments:
        - `sample`:
        """
        if isinstance(sample, int):
            sample = self.sampleNames[sample]
        sampleIdx = np.where(
            np.array([x.startswith(sample) for x in self.sampleNames]))[0][0]
        sample = self.sampleNames[sampleIdx]
        import pylab
        import mylib
        if ax is None:
            ax = pylab.gca()

        myData = self._spectrum[["mutType", sample]]
        if self.context == 0:
            ax.pie(myData)
        else:
            ctx5p = np.array([x[:self.context] for x in myData["mutType"]])
            ctx3p = np.array([x[-self.context:] for x in myData["mutType"]])
            ctxes = list(sorted(set(ctx5p) | set(ctx3p)))
            ctxCounts = np.array([
                myData[sample][(ctx5p == c5) & (ctx3p == c3)].sum()
                for c5 in ctxes for c3 in ctxes
            ])

            allMutTypes = np.unique(
                np.array([
                    x[self.context:self.context + 1] + ">" + x[-self.context -
                                                               1:-self.context]
                    for x in myData["mutType"]
                ]))

            pieSizeFactor = 1000.0 / np.sqrt(ctxCounts.max())
            pieSizeFactor = 1000.0 / ctxCounts.max()
            for i, c5 in enumerate(ctxes):
                for j, c3 in enumerate(ctxes):
                    ctxData = myData[(ctx5p == c5) & (ctx3p == c3)]
                    myTypes = dict((x[self.context:self.context + 1] + ">" +
                                    x[-self.context - 1:-self.context], y)
                                   for x, y in ctxData)
                    pieSize = int(
                        np.sqrt(ctxData[sample].sum()) * pieSizeFactor)
                    pieSize = 1.0 * ctxData[sample].sum() * pieSizeFactor
                    if sum(myTypes.values()) > 0:
                        mylib.draw_pie(
                            [myTypes.get(mut, 0.0) for mut in allMutTypes],
                            X=i,
                            Y=j,
                            size=pieSize,
                            ax=ax,
                            colors=self.colors)
                    else:
                        log.warning("No mutations for context %sX%s sample %s"
                                    % (c5, c3, sample))
            ax.set_xticks(np.arange(len(ctxes)))
            ax.set_yticks(np.arange(len(ctxes)))
            ax.set_xticklabels(ctxes, rotation="vertical")
            ax.set_yticklabels(ctxes)
            ax.set_xlabel("5' context")
            ax.set_ylabel("3' context")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            legendPoints = []
            for mut, col in zip(allMutTypes, self.colors):
                legendPoints.append(
                    ax.scatter(
                        [-10.0], [-10.0],
                        label=mut,
                        facecolors=col,
                        edgecolors=col))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend(legendPoints, allMutTypes)

        ax.set_title("%s (%s mut.)" %
                     (self.labels[sampleIdx], "{:,}".format(ctxCounts.sum())))

        return ax

    def plotSignature(self, component=0, useXticks=True, ax=None):
        import pylab
        if ax is None:
            ax = pylab.gca()

        from matplotlib.ticker import FuncFormatter

        def to_percent(y, position):
            # Ignore the passed in position. This has the effect of scaling the default
            # tick locations.
            s = str(100 * y)

            # The percent symbol needs escaping in latex
            if matplotlib.rcParams['text.usetex'] == True:
                return s + r'$\%$'
            else:
                return s + '%'

        mutLabels = self.spectrum["mutType"][self.featPermutation]

        componentVals = self.decomp.components_[component]
        if self.decompType == "NMF":
            formatter = FuncFormatter(to_percent)
            ax.yaxis.set_major_formatter(formatter)
            componentVals /= componentVals.sum()

        ax.bar(
            np.arange(self.features),
            componentVals[self.featPermutation],
            color=self.featColors[self.featPermutation])
        ax.set_ylabel("Weight")
        ax.set_title("%dst signature of mutation spectrum" % (component + 1))
        ax.set_xticks(np.arange(self.features) + 0.4)
        if useXticks:
            ax.set_xticklabels(mutLabels, rotation="vertical")
        return ax

    def plotSignatureCtx(self, component=0, ax=None):
        "Plot given signature as context matrix of piecharts"

        import pylab
        import mylib
        if ax is None:
            ax = pylab.gca()

        myData = self.decomp.components_[component]
        if self.context == 0:
            ax.pie(myData)
        else:
            ctx5p = np.array(
                [x[:self.context] for x in self.spectrum["mutType"]])
            ctx3p = np.array(
                [x[-self.context:] for x in self.spectrum["mutType"]])
            ctxes = list(sorted(set(ctx5p) | set(ctx3p)))
            ctxCounts = np.array([
                myData[(ctx5p == c5) & (ctx3p == c3)].sum()
                for c5 in ctxes for c3 in ctxes
            ])

            allMutTypes = np.unique(
                np.array([
                    x[self.context:self.context + 1] + ">" + x[-self.context -
                                                               1:-self.context]
                    for x in self.spectrum["mutType"]
                ]))

            pieSizeFactor = 1000.0 / np.sqrt(ctxCounts.max())
            pieSizeFactor = 1000.0 / ctxCounts.max()
            for i, c5 in enumerate(ctxes):
                for j, c3 in enumerate(ctxes):
                    ctxIdxs = (ctx5p == c5) & (ctx3p == c3)
                    ctxData = myData[ctxIdxs]
                    myTypes = dict((x[self.context:self.context + 1] + ">" +
                                    x[-self.context - 1:-self.context], y)
                                   for x, y in zip(self.spectrum["mutType"][
                                       ctxIdxs], ctxData))

                    pieSize = 1.0 * ctxData.sum() * pieSizeFactor
                    if sum(myTypes.values()) > 0:
                        mylib.draw_pie(
                            [myTypes.get(mut, 0.0) for mut in allMutTypes],
                            X=i,
                            Y=j,
                            size=pieSize,
                            ax=ax,
                            colors=self.colors)
                    else:
                        log.warning("Weight for context %sX%s " % (c5, c3))
            ax.set_xticks(np.arange(len(ctxes)))
            ax.set_yticks(np.arange(len(ctxes)))
            ax.set_xticklabels(ctxes, rotation="vertical")
            ax.set_yticklabels(ctxes)
            ax.set_xlabel("5' context")
            ax.set_ylabel("3' context")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            legendPoints = []
            for mut, col in zip(allMutTypes, self.colors):
                legendPoints.append(
                    ax.scatter(
                        [-10.0], [-10.0],
                        label=mut,
                        facecolors=col,
                        edgecolors=col))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend(legendPoints, allMutTypes)

        ax.set_title("%dth Signature" % (component + 1))

        return ax

    def proportionExposure(self):
        """Scale components and transformed data such that
        components K[:,i] sum to one and

        V = N*K*M + e

        where V is the mutation spectrum, N is the number of mutations in samples,
        K is the components, M is the (scaled) exposures and e is the residual.

        return (K,M)
        """
        K = (self.decomp.components_.T / self.decomp.components_.sum(axis=1)).T
        M = (self.transformed * self.decomp.components_.sum(axis=1))
        M = (M.T / self.V.sum(axis=0)).T
        return (K, M)

    def plotTransformed(self,
                        ij=(0, 1),
                        withPies=False,
                        scaleTransform=False,
                        ax=None):
        """Plot samples transformed in two coordinates.
        Scale contributions + residuals to sum to one for each sample if scaleTransform.

        """
        transformed = self.transformed
        if scaleTransform:
            components, transformed = self.proportionExposure()

        import pylab
        if ax is None:
            ax = pylab.gca()
        if withPies:
            import mylib
            pieSize = np.sqrt(1.0 * self.Ns) / np.pi
            log.info(str(pieSize))

            for i in range(self.samples):
                mylib.draw_pie(
                    self.V[:, i],
                    transformed[i, ij[0]],
                    transformed[i, ij[1]],
                    labels=self.spectrum["mutType"],
                    size=int(pieSize[i]),
                    ax=ax,
                    colors=self.colors)

        else:
            ax.scatter(transformed[:, ij[0]], transformed[:, ij[1]])
        if scaleTransform:
            ax.set_ylabel("Proportion of %d:th signature" % (ij[1] + 1))
            ax.set_xlabel("Proportion of %d:th signature" % (ij[0] + 1))
        else:
            ax.set_ylabel("Weight of %d:th signature" % (ij[1] + 1))
            ax.set_xlabel("Weight of %d:th signature" % (ij[0] + 1))

        ax.set_title("Tumor samples by mutation signatures")
        Ytol = max(ax.get_ylim()) / 20.0
        Xtol = max(ax.get_xlim()) / 20.0

        af = AnnoteFinder(
            transformed[:, ij[0]],
            transformed[:, ij[1]],
            self.labels,
            ytol=Ytol,
            xtol=Xtol,
            axis=ax)
        pylab.connect('button_press_event', af)
        return ax

    def writeTransformed(self, outFname):
        """Write transformed data

        Arguments:
        - `outFname`:
        """
        import csv

        o = csv.writer(open(outFname, "w"), delimiter="\t")
        o.writerow(["sampleName"] +
                   ["exposure%d" % (i) for i in range(self.rank)])
        for sn, tran in zip(self.sampleNames, self.transformed):
            o.writerow([sn] + tran.tolist())

    def writeSignatures(self, outFname):
        """Write transformed data

        Arguments:
        - `outFname`:
        """
        import csv
        o = csv.writer(open(outFname, "w"), delimiter="\t")
        o.writerow(["mutation"] + ["weight%d" % (i) for i in range(self.rank)])
        import logging as log
        log.info("Rank %d shape %s" %
                 (self.rank, str(self.decomp.components_.shape)))

        assert self.rank == self.decomp.components_.shape[0]
        #assert np.allclose(self.spectrum.components_,self.decomp.components_.T),"Something fishy with components."
        for sn, tran in zip(self.spectrum["mutType"],
                            self.decomp.components_.T):
            o.writerow([sn] + tran.tolist())


def parseSeqContextArgs():
    import logging as log
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and visualise sequence context signatures output from vcfSiteSequenceContext.py"
    )

    parser.add_argument(
        "-i",
        "--input",
        action="append",
        help="Input matrix file",
        required=True)
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=3,
        help="Number of signatures to extract [default: %(default)s]", )
    parser.add_argument(
        "-c",
        "--context",
        type=int,
        default=1,
        help="Context length [default: %(default)s]", )
    parser.add_argument(
        "-C",
        "--clearContext",
        default=False,
        action="store_true",
        help="Remove the context from spectrum [default: %(default)s]", )

    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="NMF",
        help="Method for decomposition (NMF|PCA|ICA|sparseNMF). If this is the pickle file previously produced with --findAlleSignatures, load it and only do postprocessing [default: %(default)s]",
    )

    parser.add_argument(
        "-A",
        "--findAllSignatures",
        type=str,
        help="Find decomposition signatures by bootstrapping and averaging with ranks from 2 to 'rank' inclusive. Store the resulting signatures to file given as argument. Second file, with .json suffix, contains the QC values from the analysis.[default: %(default)s]",
    )

    parser.add_argument(
        "-W",
        "--output",
        type=str,
        default=None,
        help="Write tsv files with signature exposures and mutation weights for samples and signatures. [default: %(default)s]",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Only consider samples named in the given file. [default: %(default)s]",
    )

    parser.add_argument(
        "-B",
        "--bootsamples",
        type=int,
        default=150,
        help="Number of bootstrap samples to draw [default: %(default)s]", )
    parser.add_argument(
        "--bootIndividuals",
        default=False,
        action="store_true",
        help="In addition to Alexandrov et.al (2013) also bootstrap the input individuals [default: %(default)s]",
    )
    parser.add_argument(
        "--SHUFFLE",
        default=False,
        action="store_true",
        help="Shuffle the mutation counts between the individuals independently for each mutation context. [default: %(default)s]",
    )
    
    
    parser.add_argument(
        "--plotSample",
        type=str,
        action="append",
        default=[],
        help="Plot signature of a given sample [default: %(default)s]", )

    parser.add_argument(
        "-p",
        "--plot",
        default=[],
        action="append",
        help="Plot signatures and sample projections [default: %(default)s]", )
    parser.add_argument(
        "-s",
        "--SNVs",
        default=False,
        action="store_true",
        help="Consider only single nucleotide variants [default: %(default)s]",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Even more verbose output [default: %(default)s]",
    )
    args = parser.parse_args()

    if args.debug:
        log.basicConfig(level=log.DEBUG)
    else:
        log.basicConfig(level=log.INFO)    
        
    log.info(" ".join(sys.argv))
    log.info(args)
    return args


if __name__ == '__main__':



    args = parseSeqContextArgs()
    import logging as log
    


    import numpy as np

    fullSpectrum = None
    for fname in args.input:
        sp = np.genfromtxt(fname, names=True, dtype=None, encoding=None)
        if fullSpectrum is None:
            fullSpectrum = sp
        else:
            import mylib
            fullSpectrum = mylib.mergeStructArrays(fullSpectrum, sp, "mutType")

    if args.subset is not None:
        import re
        log.info("Filtering for samples in %s" % (args.subset))
        pStr = "|".join(x.strip() for x in open(args.subset)
                        if len(x.strip()) > 1)
        pat = re.compile("^(%s).*$" % (pStr))
        cols = ["mutType"] + [
            x for x in fullSpectrum.dtype.names
            if pat.match(x) is not None and x != "mutType"
        ]
        log.info("Keeping %d samples out of %d" %
                 (len(cols) - 1, len(fullSpectrum.dtype.names) - 1))
        fullSpectrum = fullSpectrum[cols]

    if args.SNVs:
        # Only use single nucleotide substitutions.
        spectrum = fullSpectrum[np.fromiter((len(
            x) == 7 for x in fullSpectrum["mutType"]), bool), ]
    else:
        spectrum = fullSpectrum

    log.info("Spectrum shape: %s", str(spectrum.shape))
    assert len(spectrum["mutType"]) == len(
        set(spectrum["mutType"])), "Duplicate mutation types not allowed"

    if args.clearContext:
        spectrum = subSumContext(spectrum, args.context)

    mySpectrum = Spectrum(spectrum, args.context)
    

    if args.SHUFFLE:
        log.info("Shuffling the mutation spectrum! Output is random!!!")
        mySpectrum.shuffle()

     
    import os.path

    if args.method.upper() == "PCA":
        log.info("Running PCA")
        mySpectrum.PCA(args.rank)
    elif args.method.upper() == "NMF":
        log.info("Running NMF")
        from sklearn import decomposition
        mySpectrum.decompFun= lambda rank:decomposition.NMF(n_components=rank,max_iter=2000)
        mySpectrum.NMF(args.rank)

    elif args.method.upper() == "SPARSENMF":
        log.error("Running sparseNMF. Failed!!! just couldn't do it")
        from sklearn import decomposition
        mySpectrum.decompFun= lambda rank:decomposition.NMF(n_components=rank,max_iter=2000,sparseness="components")
        mySpectrum.NMF(args.rank, sparseness="components")
    elif args.method.upper() == "ICA":
        log.info("Running ICA")
        from sklearn import decomposition
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.base import BaseEstimator

        #mySpectrum.decompFun= lambda rank:decomposition.FastICA(n_components=rank,max_iter=2000)
        class CastTransform(BaseEstimator):
            def __init__(self, dtype=float):
                self.dtype = dtype

            def fit(self, X, *args, **kwargs):
                return self

            def transform(self, X, *args, **kwargs):
                return np.array(X, dtype=self.dtype)

        class EstimatorPipeline(Pipeline):
            def __getattr__(self, X):
                for s in self.steps[::-1]:
                    if hasattr(s[1], X):
                        #log.info("Found %s.%s"%(str(s),str(X)))
                        return getattr(s[1], X)
                raise AttributeError("Attribute %s not in %s" %
                                     (str(X), str(self)))

        log.warning("ICA based model selection is somewhat broken")
        mySpectrum.decompFun= lambda rank: EstimatorPipeline([('floatCast',CastTransform(float)),
                                                              ('scale',StandardScaler()),
                                                              ('fastICA',decomposition.FastICA(n_components=rank,max_iter=2000))])

        mySpectrum.decompFun = None
        mySpectrum.ICA(args.rank)
    elif os.path.exists(args.method):
        signatures = Signature.load(args.method)
        mySigs = [x for x in signatures if x.components_.shape[0] == args.rank]
        assert len(
            mySigs
        ) == 1, "The input pickle must have exactly 1 decomposition with %d signatures" % (
            args.rank)
        mySpectrum.setSignatures(mySigs[0])
    else:
        log.warning("No decomposition method given [%s]" % (str(args.method)))

    if args.findAllSignatures is not None:
        mySpectrum.decomp = None
        mySpectrum.findAllSignatures(
            maxBoots=args.bootsamples,
            outFname=args.findAllSignatures,
            bootIndividuals=args.bootIndividuals,
            maxRank=args.rank)
        mySigs = [
            x for x in mySpectrum.signatures
            if x.components_.shape[0] == args.rank
        ]
        assert len(
            mySigs
        ) == 1, "The input pickle must have exactly 1 decomposition with %d signatures" % (
            args.rank)
        mySpectrum.setSignatures(mySigs[0])

    if args.output is not None:
        mySpectrum.writeSignatures(args.output + ".weights.tsv")
        mySpectrum.writeTransformed(args.output + ".exposures.tsv")
    if len(args.plot) > 0:
        log.info("Plotting")
        import pylab
        if "counts" in args.plot or "all" in args.plot:
            fig = pylab.figure()
            log.info("Plotting counts")
            mySpectrum.plotCounts(fig.gca())

        if "signature" in args.plot or "all" in args.plot:
            fig, axs = pylab.subplots(args.rank, 1, sharex=True)

            for rank in range(args.rank):
                #ax=fig.add_subplot(args.rank,1,rank+1)
                mySpectrum.plotSignature(
                    component=rank,
                    ax=axs[rank],
                    useXticks=rank == (args.rank - 1))

        if "signatureContext" in args.plot or "all" in args.plot:
            for rank in range(args.rank):
                fig = pylab.figure()
                mySpectrum.plotSignatureCtx(component=rank, ax=fig.gca())

        if "spectra" in args.plot or "all" in args.plot:
            log.info("Plotting spectra")
            spectrumFig = pylab.figure()

            ax = mySpectrum.plotSpectrum(ax=pylab.gca())

        if "transformed" in args.plot or "all" in args.plot:
            log.info("Plotting transformed points")
            #    transformedFig=pylab.figure()
            #    nPlots = args.rank*(args.rank-1)/2
            fig, axs = pylab.subplots(
                args.rank - 1, args.rank - 1, sharex=False, sharey=False)
            for i in range(args.rank):
                for j in range(i + 1, args.rank):
                    #ax=transformedFig.add_subplot(args.rank,args.rank,i*args.rank+j)
                    mySpectrum.plotTransformed(
                        ij=(i, j),
                        withPies=(mySpectrum.features < 10),
                        ax=axs[j - 1, i],
                        scaleTransform="scale" in args.plot)
        import re
        tPat = re.compile("transformed:([0-9]+):([0-9]+)")
        for t in [tPat.match(x) for x in args.plot]:
            if t is None:
                continue
            pylab.figure()
            log.info("Plotting transformed axis %s" % (str(t.groups())))
            mySpectrum.plotTransformed(
                ij=(int(t.group(1)), int(t.group(2))), withPies=False)

        if "pieMatrix" in args.plot or "all" in args.plot:
            log.info("Plotting spectral pie matrix")
            yPlots = np.sqrt(2.0 * mySpectrum.samples / 3.0)
            xPlots = int(np.ceil(3.0 / 2.0 * yPlots))
            yPlots = int(np.ceil(yPlots))
            fig, axs = pylab.subplots(yPlots, xPlots)
            #        mutColors = ['red','blue','green','yellow','magenta','purple']
            for idx, s in enumerate(mySpectrum.sampleNames):
                i = idx / xPlots
                j = idx % xPlots
                mySpectrum.plotPie(i * xPlots + j, title=True, ax=axs[i, j])

            for idx in range(idx, xPlots * yPlots):
                i = idx / xPlots
                j = idx % xPlots
                axs[i, j].set_frame_on(False)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
            mySpectrum.plotPie(0, title=False, ax=axs[yPlots - 1, xPlots - 1])
            axs[yPlots - 1, xPlots - 1].set_xlim([100, 100.1])
            axs[yPlots - 1, xPlots - 1].set_xlim([100, 100.1])
            axs[yPlots - 1, xPlots - 1].legend(
                mySpectrum.spectrum["mutType"])  #,colors=mutColors)

    for pSample in args.plotSample:
        log.info("Plotting spectrum from sample %s" % (pSample))
        fig = pylab.figure()
        mySpectrum.plotSpectrumSample(pSample, fig.gca())
        pylab.savefig("%s.png" % (pSample))
        pylab.savefig("%s.pdf" % (pSample))

    if args.plot or len(args.plotSample) > 0:
        log.info("Showing")
        pylab.show()

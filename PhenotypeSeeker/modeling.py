#!/usr/bin/python2.7

__author__ = "Erki Aun"
__version__ = "0.3.0"
__maintainer__ = "Erki Aun"
__email__ = "erki.aun@ut.ee"

from itertools import chain, izip, izip_longest, permutations
from subprocess import call, Popen, PIPE, check_output
import math
import sys
import warnings
import time
warnings.showwarning = lambda *args, **kwargs: None

from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from cogent import LoadTree
from cogent.align.weights.methods import GSC
from collections import Counter, OrderedDict
from multiprocess import Manager, Pool, Value
from scipy import stats
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (Lasso, LogisticRegression, Ridge, ElasticNet,
    SGDClassifier)
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, r2_score, mean_squared_error, recall_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, accuracy_score
    )
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from functools import partial
import Bio
import sklearn.datasets
import numpy as np


class process_input():

    samples = OrderedDict()
    phenotypes_to_analyse = OrderedDict()
    pool = None
    lock = None
    
    @classmethod
    def get_input_data(cls, inputfilename, take_logs):
        # Read the data from inputfile into "samples" directory
        samples = OrderedDict()
        Samples.take_logs = take_logs
        with open(inputfilename) as inputfile:
            for i, line in enumerate(inputfile):
                if i == 0:
                    firstline = line.split()
                    Samples.no_phenotypes = len(firstline)-2
                    if firstline[0] == "SampleID":
                        Samples.phenotypes = firstline[2:]
                        Samples.headerline = True
                        continue
                    else:
                        for j in xrange(1, Samples.no_phenotypes + 1):
                            Samples.phenotypes.append("phenotype%s" %j)
                sample_name = line.split()[0]
                cls.samples[sample_name] = (
                    Samples.from_inputfile(line)
                    )

    # ---------------------------------------------------------
    # Set parameters for multithreading
    @classmethod
    def get_multithreading_parameters(cls):
        cls.lock = Manager().Lock()
        cls.pool = Pool(Samples.num_threads)

    # ---------------------------------------------------------
    # Functions for processing the command line input arguments

    @classmethod
    def process_input_args(
            cls, alphas, alpha_min, alpha_max, n_alphas,
            gammas, gamma_min, gamma_max, n_gammas, 
            min_samples, max_samples, mpheno, kmer_length,
            cutoff, num_threads
            ):
        Samples.alphas = cls._get_alphas(alphas, alpha_min, alpha_max, n_alphas)
        Samples.gammas = cls._get_gammas(gammas, gamma_min, gamma_max, n_gammas)
        Samples.min_samples, Samples.max_samples = cls._get_min_max(min_samples, max_samples)
        cls._get_phenotypes_to_analyse(mpheno)
        Samples.kmer_length = kmer_length
        Samples.cutoff = cutoff
        Samples.num_threads = num_threads
        
    @staticmethod
    def _get_alphas(alphas, alpha_min, alpha_max, n_alphas):       
        # Generating the vector of alphas (hyperparameters in regression analysis)
        # based on the given command line arguments.
        if alphas == None:
            alphas = np.logspace(
                math.log10(alpha_min),
                math.log10(alpha_max), num=n_alphas)
        else: 
            alphas = np.array(alphas)
        return alphas

    @staticmethod
    def _get_gammas(gammas, gamma_min, gamma_max, n_gammas):
        # Generating the vector of alphas (hyperparameters in regression analysis)
        # based on the given command line arguments.
        if gammas == None:
            gammas = np.logspace(
                math.log10(gamma_min),
                math.log10(gamma_max), num=n_gammas)
        else: 
            gammas = np.array(gammas)
        return gammas

    @staticmethod
    def _get_min_max(min_samples, max_samples):
        # Set the min and max arguments to default values
        min_samples = int(min_samples)
        if min_samples == 0:
            min_samples = 2
        max_samples = int(max_samples)
        if max_samples == 0:
            max_samples = Samples.no_samples - 2
        return min_samples, max_samples

    @classmethod
    def _get_phenotypes_to_analyse(cls, mpheno):
        if not mpheno:
            phenotypes_to_analyze = range(Samples.no_phenotypes)
        else: 
            phenotypes_to_analyze = map(lambda x: x-1, mpheno)
        for item in phenotypes_to_analyze:
            cls.phenotypes_to_analyse[Samples.phenotypes[item]] = phenotypes(Samples.phenotypes[item])

class phenotypes():

    def __init__(self, name):
        self.name = name
        self.pvalues = None

class Samples():

    phenotype_scale = "binary"
    no_samples = 0
    no_phenoypes = 0
    phenotypes = []
    take_logs = None
    headerline = None

    kmer_length = None
    cutoff = None
    alphas = None
    gammas = None
    min_samples = None
    max_samples = None
    num_threads = None

    def __init__(self, name, address, phenotypes, weight=1):
        self.name = name
        self.address = address
        self.phenotypes = phenotypes
        self.weight = weight

        Samples.no_samples += 1

    def get_kmer_lists(self):
        # Makes "K-mer_lists" directory where all lists are stored.
        # Generates k-mer lists for every sample in names_of_samples variable 
        # (list or dict).
        call(["mkdir", "-p", "K-mer_lists"])
        call(
            ["glistmaker " + self.address + " -o K-mer_lists/" 
            + self.name + " -w " + self.kmer_length + " -c " + self.cutoff], 
            shell=True
            )
        process_input.lock.acquire()
        stderr_print.currentSampleNum.value += 1
        process_input.lock.release()
        stderr_print.print_progress()

    def map_samples(self):
        # Takes k-mers, which passed frequency filtering as 
        # feature space and maps samples k-mer list to that 
        # feature space. A vector of k-mers frequency information 
        # is created for every sample.
        outputfile = "K-mer_lists/" + self.name + "_mapped.txt"
        with open(outputfile, "w+") as outputfile:
            call(
                [
                "glistquery", "K-mer_lists/" + self.name + "_" + self.kmer_length +
                ".list", "-l", "K-mer_lists/feature_vector_" + self.kmer_length +
                ".list"
                ]
                , stdout=outputfile)
        process_input.lock.acquire()
        stderr_print.currentSampleNum.value += 1
        process_input.lock.release()
        stderr_print.print_progress()

    @classmethod
    def from_inputfile(cls, line):
        phenotypes = {}
        name, address, phenotype_list = \
            line.split()[0], line.split()[1], line.split()[2:]
        if not all(x == "0" or x == "1" or x == "NA" for x in phenotypes):
            cls.phenotype_scale = "continuous"
        if cls.take_logs:
            phenotype_list = map(lambda x: math.log(x, 2), phenotypes)
        for i,j in zip(cls.phenotypes, phenotype_list):
            phenotypes[i] = j 
        return cls(name, address, phenotypes)

    @classmethod
    def get_feature_vector(cls):
        glistmaker_args = ["glistmaker"] + \
            [sample.address for sample in process_input.samples.values()] + \
            [
            '-c', cls.cutoff, '-w', cls.kmer_length, '-o', 'K-mer_lists/feature_vector'
            ]
        call(glistmaker_args)


    # -------------------------------------------------------------------
    # Functions for calculating the mash distances and GSC weights for
    # input samples.
    
    @classmethod
    def get_weights(cls):
        cls._mash_caller()
        cls._mash_output_to_distance_matrix(process_input.samples.keys(), "mash_distances.mat")
        dist_mat = cls._distance_matrix_modifier("distances.mat")
        cls._distance_matrix_to_phyloxml(process_input.samples.keys(), dist_mat)   
        cls._phyloxml_to_newick("tree_xml.txt")
        sys.stderr.write("Calculating the Gerstein Sonnhammer Coathia " \
            "weights from mash distance matrix...")
        weights = cls._newick_to_GSC_weights("tree_newick.txt")
        for key, value in weights.iteritems():
            process_input.samples[key].weight = value
    
    @classmethod
    def _mash_caller(cls):
        #Estimating phylogenetic distances between samples using mash
        sys.stderr.write("\nEstimating the Mash distances between samples...\n")
        mash_args = ["mash", "sketch", "-o", "reference", "-m", cls.cutoff]
        for sample_data in process_input.samples.values():
            mash_args.append(sample_data.address)
        process = Popen(mash_args, stderr=PIPE)
        for line in iter(process.stderr.readline, ''):
            stderr_print(line.strip())
        stderr_print("")
        with open("mash_distances.mat", "w+") as f1:
            call(["mash", "dist", "reference.msh", "reference.msh"], stdout=f1)

    @classmethod
    def _mash_output_to_distance_matrix(cls, names_of_samples, mash_distances):
        with open(mash_distances) as f1:
            with open("distances.mat", "w+") as f2:
                counter = 0
                f2.write(names_of_samples[counter])
                for line in f1:
                    distance = line.split()[2]
                    f2.write("\t" + distance)
                    counter += 1
                    if counter%cls.no_samples == 0:
                        if counter != cls.no_samples**2:
                            f2.write(
                                "\n" + names_of_samples[counter/cls.no_samples]
                                )

    @staticmethod
    def _distance_matrix_modifier(distance_matrix):
        # Modifies distance matrix to be suitable argument 
        # for Bio.Phylo.TreeConstruction._DistanceMatrix function
        distancematrix = []
        with open(distance_matrix) as f1:
            counter = 2
            for line in f1:
                line = line.strip().split()
                distancematrix.append(line[1:counter])
                counter += 1
        for i in range(len(distancematrix)):
            for j in range(len(distancematrix[i])):
                distancematrix[i][j] = float(distancematrix[i][j])
        return(distancematrix)

    @staticmethod
    def _distance_matrix_to_phyloxml(samples_order, distance_matrix):
        #Converting distance matrix to phyloxml
        dm = _DistanceMatrix(samples_order, distance_matrix)
        tree_xml = DistanceTreeConstructor().nj(dm)
        with open("tree_xml.txt", "w+") as f1:
            Bio.Phylo.write(tree_xml, f1, "phyloxml")

    @staticmethod
    def _phyloxml_to_newick(phyloxml):
        #Converting phyloxml to newick
        with open("tree_newick.txt", "w+") as f1:
            Bio.Phylo.convert(phyloxml, "phyloxml", f1, "newick")

    @staticmethod
    def _newick_to_GSC_weights(newick_tree):
        # Calculating Gerstein Sonnhammer Coathia weights from Newick 
        # string. Returns dictionary where sample names are keys and GSC 
        # weights are values.
        tree=LoadTree(newick_tree)
        weights=GSC(tree)
        for item in weights:
            weights[item] = 1 - weights[item]
        return(weights)


class stderr_print():
    # --------------------------------------------------------
    # Functions and variables necessarry to show the progress 
    # information in standard error.

    currentSampleNum = Value("i", 0)
    currentKmerNum = Value("i", 0)
    previousPercent = Value("i", 0)

    def __init__(self,data):
        sys.stderr.write("\r\x1b[K"+data.__str__())
        sys.stderr.flush()

    @classmethod
    def check_progress(cls, totalKmers, text, phenotype=""):
        currentPercent = (cls.currentKmerNum.value/float(totalKmers))*100
        if int(currentPercent) > cls.previousPercent.value:
            output = "\t" + phenotype + "%d%% of " % (
                currentPercent
                ) + text
            cls.previousPercent.value = int(currentPercent)
            cls(output)

    @classmethod
    def print_progress(cls):
        output = "\t%d of %d lists generated." % (
            cls.currentSampleNum.value, Samples.no_samples
            )
        cls(output)

class metrics():
    # ---------------------------------------------------------
    # Self-implemented performance measure functions

    @staticmethod
    def VME(targets, predictions):
        # Function to calculate the very major error (VME) rate
        VMEs = 0
        for item in zip(targets, predictions):
            if item[0] == 1 and item[1] == 0:
                VMEs += 1
        VME = str(float(VMEs)/len(targets)*100)+"%"
        return VME

    @staticmethod
    def ME(targets, predictions):
        # Function to calculate the major error (ME) rate
        MEs = 0
        for item in zip(targets, predictions):
            if item[0] == 0 and item[1] == 1:
                 MEs += 1
        ME = str(float(MEs)/len(targets)*100)+"%"
        return ME

    @staticmethod
    def within_1_tier_accuracy(targets, predictions):
        # Calculate the plus/minus one dilution factor accuracy
        # for predicted antibiotic resistance values.
        within_1_tier = 0
        for item in zip(targets, predictions):
            if abs(item[0]-item[1]) <= 1:
                within_1_tier +=1
        accuracy = float(within_1_tier)/len(targets)
        return accuracy

class kmers():

    phenotype_instances  = OrderedDict()

    vectors_as_multiple_input = []
    progress_checkpoint = Value("i", 0)
    no_kmers_to_analyse = Value("i", 0)

    # -------------------------------------------------------------------
    # Functions for calculating the association test results for kmers.
    @classmethod
    def test_kmers_association_with_phenotype(cls):
        if Samples.phenotype_scale == "continuous":
            sys.stderr.write("\nConducting the k-mer specific Welch t-tests:\n")
        else:
            sys.stderr.write("\nConducting the k-mer specific chi-square tests:\n")
        cls.get_params_for_kmers_testing()
        for phenotype in process_input.phenotypes_to_analyse.keys():
            stderr_print.currentKmerNum.value = 0
            stderr_print.previousPercent.value = 0
            pvalues_from_all_threads = process_input.pool.map(partial(
                    cls.get_kmers_tested, phenotype
                    ), 
                cls.vectors_as_multiple_input
                )
            process_input.phenotypes_to_analyse[phenotype].pvalues = \
                list(chain(*pvalues_from_all_threads))
            sys.stderr.write("\n")
            cls.concatenate_test_files(phenotype)

    @classmethod
    def get_params_for_kmers_testing(cls):
        cls._split_sample_vectors_for_multithreading()
        cls._splitted_vectors_to_multiple_input()
        cls.no_kmers_to_analyse.value = int(
            check_output(
                ['wc', '-l', "K-mer_lists/" + process_input.samples.keys()[0] + "_mapped.txt"]
                ).split()[0]
            )
        cls.progress_checkpoint.value = int(math.ceil(cls.no_kmers_to_analyse.value/(100*Samples.num_threads)))

    @staticmethod
    def _split_sample_vectors_for_multithreading():
        for sample in process_input.samples:
            call([
                "split -a 5 -d -n r/" + str(Samples.num_threads) + " K-mer_lists/" +
                sample + "_mapped.txt " + "K-mer_lists/" + sample + "_mapped_"
                ], shell=True)

    @classmethod
    def _splitted_vectors_to_multiple_input(cls):
        vectors_as_multiple_input = []
        for i in range(Samples.num_threads):
            cls.vectors_as_multiple_input.append(["K-mer_lists/" + sample + "_mapped_%05d" %i for sample in process_input.samples])

    @classmethod
    def get_kmers_tested(cls, phenotype, split_of_kmer_lists):

        names_of_samples = process_input.samples.keys()
        phenotypes_of_samples = [sample_data.phenotypes[phenotype] for sample_data in process_input.samples.values()]
        pvalues = []
        counter = 0

        multithreading_code = split_of_kmer_lists[0][-5:]
        test_results_file = open(cls.test_result_output(
            phenotype, multithreading_code
            ), "w")
        text1_4_stderr = cls.get_text1_4_stderr(phenotype)
        text2_4_stderr = "tests conducted."
        for line in izip_longest(*[open(item) for item in split_of_kmer_lists], fillvalue = ''):
            counter += 1
            if counter%cls.progress_checkpoint.value == 0:
                process_input.lock.acquire()
                stderr_print.currentKmerNum.value += cls.progress_checkpoint.value
                process_input.lock.release()
                stderr_print.check_progress(
                    cls.no_kmers_to_analyse.value, text2_4_stderr, text1_4_stderr
                )
            kmer = line[0].split()[0]
            kmer_presence = [j.split()[1].strip() for j in line]

            if Samples.phenotype_scale == "binary":
                pvalue = cls.conduct_chi_squared_test(
                    phenotypes_of_samples, names_of_samples, kmer, kmer_presence,
                    test_results_file
                    )
            elif Samples.phenotype_scale == "continuous":
                pvalue = cls.conduct_t_test(
                    phenotypes_of_samples, names_of_samples, kmer, kmer_presence,
                    test_results_file
                    )
            if pvalue:
                pvalues.append(pvalue)
        process_input.lock.acquire()
        stderr_print.currentKmerNum.value += counter%cls.progress_checkpoint.value
        process_input.lock.release()
        stderr_print.check_progress(
            cls.no_kmers_to_analyse.value, text2_4_stderr, text1_4_stderr
        )
        test_results_file.close()
        return(pvalues)

    @staticmethod
    def test_result_output(phenotype, code):
        if Samples.phenotype_scale == "continuous":
            beginning_text = "t-test_results_"
        else:
            beginning_text = "chi-squared_test_results_"
        if Samples.headerline:
            outputfile = beginning_text + \
                phenotype + "_" + code + ".txt"
        elif len(process_input.phenotypes_to_analyse) > 1:
            outputfile = beginning_text + \
                phenotype + "_" + code + ".txt"
        else:
            outputfile = beginning_text + code + ".txt"
        return outputfile

    @staticmethod
    def get_text1_4_stderr(phenotype):
        if Samples.headerline:
            text1_4_stderr = phenotype + ": "
        elif len(process_input.phenotypes_to_analyse) > 1:
            text1_4_stderr = "phenotype " + phenotype + ": "
        else:
            text1_4_stderr = ""
        return text1_4_stderr

    @classmethod
    def conduct_t_test(
        cls, phenotypes_of_samples, names_of_samples, kmer, kmer_presence, 
        test_results_file
        ):
        samples_w_kmer = []
        x = []
        y = []
        x_weights = []
        y_weights = []
        
        cls.get_samples_distribution_for_ttest(
            x, y, x_weights, y_weights, kmer_presence,
            samples_w_kmer, phenotypes_of_samples, names_of_samples
            )

        if len(x) < Samples.min_samples or len(y) < 2 or len(x) > Samples.max_samples:
            return

        t_statistic, pvalue, mean_x, mean_y = t_test(
            x, y, x_weights, y_weights
            )

        test_results_file.write(
            kmer + "\t" + str(round(t_statistic, 2)) + "\t" + \
            "%.2E" % pvalue + "\t" + str(round(mean_x, 2)) + "\t" + \
            str(round(mean_y,2)) + "\t" + str(len(samples_w_kmer)) + "\t| " + \
            " ".join(samples_w_kmer) + "\n"
            )
        return pvalue

    @staticmethod
    def get_samples_distribution_for_ttest(
            x, y, x_weights, y_weights, list1,
            samples_w_kmer, phenotypes_of_samples, names_of_samples
            ):
        for i, item in enumerate(phenotypes_of_samples):
            sample_name = names_of_samples[i]
            if item != "NA":
                if list1[i] == "0":
                    y.append(float(item))
                    y_weights.append(process_input.samples[sample_name].weight)
                else:
                    x.append(float(item))
                    x_weights.append(process_input.samples[sample_name].weight)
                    samples_w_kmer.append(sample_name)

    @staticmethod
    def t_test(x, y, x_weights, y_weights):
        #Parametes for group containig the k-mer
        wtd_mean_y = np.average(y, weights=y_weights)
        sumofweightsy = sum(y_weights)
        ybar = np.float64(sum([i*j for i,j in zip(y, y_weights)])/sumofweightsy)
        vary = sum([i*j for i,j in zip(y_weights, (y - ybar)**2)])/(sumofweightsy-1)
        
        #Parameters for group not containig the k-mer
        wtd_mean_x = np.average(x, weights=x_weights)
        sumofweightsx = sum(x_weights)
        xbar = np.float64(sum([i*j for i,j in zip(x, x_weights)])/sumofweightsx)
        varx = sum([i*j for i,j in zip(x_weights, (x - xbar)**2)])/(sumofweightsx-1)

        #Calculating the weighted Welch's t-test results
        dif = wtd_mean_x-wtd_mean_y
        sxy = math.sqrt((varx/sumofweightsx)+(vary/sumofweightsy))
        df = (((varx/sumofweightsx)+(vary/sumofweightsy))**2)/((((varx/sumofweightsx)**2)/(sumofweightsx-1))+((vary/sumofweightsy)**2/(sumofweightsy-1)))
        t= dif/sxy
        pvalue = stats.t.sf(abs(t), df)*2

        return t, pvalue, wtd_mean_x, wtd_mean_y

    @classmethod
    def conduct_chi_squared_test(
        cls, phenotypes_of_samples, names_of_samples, kmer, kmer_presence,
        test_results_file
        ):
        samples_w_kmer = []
        (
        w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer
        ) = cls.get_samples_distribution_for_chisquared(
            phenotypes_of_samples, names_of_samples, kmer_presence, samples_w_kmer
            )
        (w_pheno, wo_pheno, w_kmer, wo_kmer, total) = cls.get_totals_in_classes(
            w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer
            )
        no_samples_w_kmer = len(samples_w_kmer)
        if no_samples_w_kmer < Samples.min_samples or no_samples_w_kmer > Samples.max_samples:
            return

        (
        w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected,
        wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected
        ) = cls.get_expected_distribution(
            w_pheno, wo_pheno, w_kmer, wo_kmer, total)

        chisquare_results = stats.chisquare(
            [
            w_pheno_w_kmer, w_pheno_wo_kmer,
            wo_pheno_w_kmer, wo_pheno_wo_kmer
            ],
            [
            w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected, 
            wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected
            ],
            1
            )
        test_results_file.write(
            kmer + "\t%.2f\t%.2E\t" % chisquare_results 
            + str(no_samples_w_kmer)  +"\t| " + " ".join(samples_w_kmer) + "\n"
            )
        pvalue = chisquare_results[1]
        return pvalue

    @staticmethod
    def get_samples_distribution_for_chisquared(
            phenotypes_of_samples, names_of_samples, list1, samples_w_kmer
            ):
        with_pheno_with_kmer = 0
        with_pheno_without_kmer = 0
        without_pheno_with_kmer = 0
        without_pheno_without_kmer = 0
        for i, item in enumerate(phenotypes_of_samples):
            sample_name = names_of_samples[i]
            if item != "NA":
                if item == "1":
                    if (list1[i] != "0"):
                        with_pheno_with_kmer += process_input.samples[sample_name].weight 
                        samples_w_kmer.append(sample_name)
                    else:
                        with_pheno_without_kmer += process_input.samples[sample_name].weight
                else:
                    if (list1[i] != "0"):
                        without_pheno_with_kmer += process_input.samples[sample_name].weight
                        samples_w_kmer.append(names_of_samples[i])
                    else:
                        without_pheno_without_kmer += process_input.samples[sample_name].weight
        return(
            with_pheno_with_kmer, with_pheno_without_kmer,
            without_pheno_with_kmer, without_pheno_without_kmer
            )

    @staticmethod
    def get_totals_in_classes(
            w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer
            ):
        w_pheno = (w_pheno_w_kmer + w_pheno_wo_kmer)
        wo_pheno = (
            wo_pheno_w_kmer + wo_pheno_wo_kmer
            )
        w_kmer = (w_pheno_w_kmer + wo_pheno_w_kmer)
        wo_kmer = (
            w_pheno_wo_kmer + wo_pheno_wo_kmer
            )
        total = w_pheno + wo_pheno
        return w_pheno, wo_pheno, w_kmer, wo_kmer, total

    @staticmethod
    def get_expected_distribution(w_pheno, wo_pheno, w_kmer, wo_kmer, total):
        w_pheno_w_kmer_expected = ((w_pheno * w_kmer)
                         / float(total))
        w_pheno_wo_kmer_expected = ((w_pheno * wo_kmer) 
                          / float(total))
        wo_pheno_w_kmer_expected  = ((wo_pheno * w_kmer)
                          / float(total))
        wo_pheno_wo_kmer_expected = ((wo_pheno * wo_kmer)
                           / float(total))
        return(
            w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected,
            wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected
            )

    @staticmethod
    def concatenate_test_files():
        if Samples.phenotype_scale == "continuous":
            beginning_text = "t-test_results_"
        else:
            beginning_text = "chi-squared_test_results_"
        if Samples.headerline:
            call(
                [
                "cat " + beginning_text + phenotype + "_* > " +
                beginning_text + phenotype + ".txt"
                ],
                shell=True
                )
            for l in range(Samples.num_threads):
                call(
                    [
                    "rm " + beginning_text + phenotype +
                    "_%05d.txt" % l
                    ],
                    shell=True
                    )
        elif Samples.no_phenotypes > 1:
            call(
                [
                "cat " + beginning_text + phenotype + "_* > " +
                beginning_text + phenotype + ".txt"
                ],
                shell=True
                )
            for l in range(Samples.num_threads):
                call(
                    [
                    "rm " + beginning_text + phenotype + "_%05d.txt" %l
                    ],
                    shell=True
                    )     
        else:
            call(
                [
                "cat " + beginning_text + "* > " + beginning_text[:-1] + 
                ".txt && rm " + beginning_text + "*"
                ],
                shell=True
                )

def kmer_filtering_by_pvalue(
        pvalue, pvalues_all_phenotypes, kmer_limit,
        FDR=False, B=False
        ):
    # Filters the k-mers by their p-value achieved in statistical 
    # testing.
    sys.stderr.write("Filtering the k-mers by p-value:\n")
    kmers_passed_all_phenotypes = []
    for j, k in enumerate(Samples.phenotypes_to_analyse):
        nr_of_kmers_tested = float(len(pvalues_all_phenotypes[j]))
        stderr_print.currentKmerNum.value = 0
        stderr_print.previousPercent.value = 0
        checkpoint = int(math.ceil(nr_of_kmers_tested/100))
        counter = 0
        kmers_passed = []
        phenotype = Samples.phenotypes[k]
        if Samples.phenotype_scale == "continuous":
            test = "t-test"
        elif Samples.phenotype_scale == "binary":
            test = "chi-squared_test"
        if Samples.headerline:
            f1 = open(test + "_results_" + phenotype + ".txt")
            f2 = open(
                "k-mers_filtered_by_pvalue_" + phenotype + ".txt", "w+")
            phenotype = phenotype + ": "
        elif Samples.no_phenotypes > 1:
            f1 = open(test + "_results_" + phenotype + ".txt")
            f2 = open("k-mers_filtered_by_pvalue_" + phenotype + ".txt", "w+")
            phenotype = "phenotype " + phenotype + ": "
        else:
            f1 = open(test + "_results.txt")
            f2 = open("k-mers_filtered_by_pvalue.txt", "w+")
            phenotype = ""

        if Samples.phenotype_scale == "continuous":
            f2.write(
                "K-mer\tWelch's_t-statistic\tp-value\t+_group_mean\
                \t-_group_mean\tNo._of_samples_with_k-mer\
                \tSamples_with_k-mer\n"
                )
        elif Samples.phenotype_scale == "binary":
            f2.write(
                "K-mer\tChi-square_statistic\tp-value\
                \tNo._of_samples_with_k-mer\tSamples_with_k-mer\n"
                )
        number_of_kmers = 0
        pvalues_ascending = sorted(pvalues_all_phenotypes[j])
        max_pvalue_by_limit = float('%.2E' % pvalues_ascending[kmer_limit-1])
        if B:
            for line in f1:
                counter += 1
                max_pvalue_by_B = (
                    pvalue/nr_of_kmers_tested
                    )
                list1 = line.split()
                if float(list1[2]) < (max_pvalue_by_B):
                    f2.write(line)
                    if float(list1[2]) <= max_pvalue_by_limit:
                        kmers_passed.append(list1[0])
                        number_of_kmers += 1
                if counter%checkpoint == 0:
                    process_input.lock.acquire()
                    stderr_print.currentKmerNum.value += checkpoint
                    process_input.lock.release()
                    stderr_print.check_progress(
                        nr_of_kmers_tested, "k-mers filtered.", phenotype
                    )
        elif FDR:
            max_pvalue_by_FDR = 0
            for i, item in enumerate(pvalues_ascending):
                if  (item  < (
                        (i+1) 
                        / nr_of_kmers_tested) * pvalue
                        ):
                    highest_sign_pvalue = item
                elif item > pvalue:
                    break
            for line in f1:
                counter +=1
                list1 = line.split()
                if float(list1[2]) <= max_pvalue_by_FDR:
                    f2.write(line)
                    if float(list1[2]) <= max_pvalue_by_limit:
                        kmers_passed.append(list1[0])
                        number_of_kmers += 1
                if counter%checkpoint == 0:
                    process_input.lock.acquire()
                    stderr_print.currentKmerNum.value += checkpoint
                    process_input.lock.release()
                    stderr_print.check_progress(
                        nr_of_kmers_tested, "k-mers filtered.", phenotype
                    )
        else:
            for line in f1:
                counter += 1
                list1 = line.split()
                if  float(list1[2]) < pvalue:
                    f2.write(line)
                    if float(list1[2]) <= max_pvalue_by_limit:
                        kmers_passed.append(list1[0])
                        number_of_kmers += 1
                if counter%checkpoint == 0:
                    process_input.lock.acquire()
                    stderr_print.currentKmerNum.value += checkpoint
                    process_input.lock.release()
                    stderr_print.check_progress(
                        nr_of_kmers_tested, "k-mers filtered.", phenotype
                    )
        kmers_passed_all_phenotypes.append(kmers_passed)
        process_input.lock.acquire()
        stderr_print.currentKmerNum.value += counter%checkpoint
        process_input.lock.release()
        stderr_print.check_progress(
            nr_of_kmers_tested, "k-mers filtered.", phenotype
            )
        if len(Samples.phenotypes_to_analyse) > 1 and k != Samples.phenotypes_to_analyse[-1]:
            sys.stderr.write("\n")
        if len(kmers_passed) == 0:
            f2.write("\nNo k-mers passed the filtration by p-value.\n")
        f1.close()
        f2.close()        
    return(kmers_passed_all_phenotypes)

def get_kmer_presence_matrix(kmers_passed, split_of_kmer_lists):
    kmers_presence_matrix = []
    features = []
    
    for line in izip_longest(*[open(item) for item in split_of_kmer_lists], fillvalue = ''):
        if line[0].split()[0] in kmers_passed:
            features.append(line[0].split()[0])
            kmers_presence_matrix.append(map(
                lambda x: 0 if x == 0 else 1,
                map(int, [j.split()[1].strip() for j in line])
                ))
    return(kmers_presence_matrix, features)

def linear_regression(
	    kmer_lists_splitted,
	    kmers_passed_all_phenotypes, penalty, n_splits, testset_size,
	    l1_ratio, max_iter, tol
	    ):

    names_of_samples = process_input.samples.keys()

    # Applies linear regression with on k-mers
    # that passed the filtering by p-value of statistical test. K-mers
    # presence/absence (0/1) in samples are used as independent
    # parameters, resistance value (continuous) is used as dependent
    # parameter.
    if len(Samples.phenotypes_to_analyse) > 1:
        sys.stderr.write("\nConducting the linear regression analysis:\n")
    elif Samples.headerline:
        sys.stderr.write("\nConducting the linear regression analysis of " 
            +  Samples.phenotypes[0] + " data...\n")
    else:
        sys.stderr.write("\nConducting the linear regression analysis...\n")

    for j, k in enumerate(Samples.phenotypes_to_analyse):
        #Open files to write results of linear regression
        phenotype = Samples.phenotypes[k]
        if Samples.headerline:
            f1 = open("summary_of_lin_reg_analysis" 
                     + phenotype + ".txt", "w+")
            f2 = open("k-mers_and_coefficients_in_lin_reg_model_" 
                     + phenotype + ".txt", "w+")
            model_filename = "lin_reg_model_" + phenotype + ".pkl"
            if len(Samples.phenotypes_to_analyse) > 1:
                sys.stderr.write("\tregression analysis of " 
                    +  phenotype + " data...\n")
        elif Samples.no_phenotypes > 1:
            f1 = open("summary_of_lin_reg_analysis" + phenotype + ".txt", "w+")
            f2 = open("k-mers_and_coefficients_in_lin_reg_model_" 
                     + phenotype + ".txt", "w+")
            model_filename = "lin_reg_model_" + phenotype + ".pkl"
            sys.stderr.write("\tregression analysis of phenotype " 
                +  phenotype + " data...\n")
        else:
            f1 = open("summary_of_lin_reg_analysis.txt", "w+")
            f2 = open("k-mers_and_coefficients_in_lin_reg_model.txt", "w+")
            model_filename = "lin_reg_model.txt"

        if len(kmers_passed_all_phenotypes[j]) == 0:
            f1.write("No k-mers passed the step of k-mer selection for " \
                "regression analysis.\n")
            continue

        # Generating a binary k-mer presence/absence matrix and a list
        # of k-mer names based on information in k-mer_matrix.txt 
        matrix_and_features = map(
            list, zip(
                *process_input.pool.map(
                    partial(
                        get_kmer_presence_matrix,
                        set(kmers_passed_all_phenotypes[j])
                        ),
                    kmer_lists_splitted
                    )
                )
            )
        kmers_presence_matrix = [
            item for sublist in matrix_and_features[0] for item in sublist
            ]
        features = [
            item for sublist in matrix_and_features[1] for item in sublist
            ]
        Phenotypes = [item.phenotypes[k] for item in process_input.samples.values()]


        # Converting data into Python array formats suitable to use in
        # sklearn modeling. Also, deleting information associated with
        # stains missing the phenotype data
        features = np.array(features)
        Phenotypes = np.array(Phenotypes)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        samples_in_analyze = np.array(names_of_samples)
        to_del = []
        for i, item in enumerate(Phenotypes):
            if item == "NA":
                to_del.append(i)
        kmers_presence_matrix = np.delete(kmers_presence_matrix, to_del, 0)
        Phenotypes = map(float, np.delete(Phenotypes, to_del, 0))            
        samples_in_analyze = np.delete(samples_in_analyze, to_del, 0)

        # Insert data into linear regression dataset 
        dataset = sklearn.datasets.base.Bunch(
        	data=kmers_presence_matrix, target=Phenotypes,
        	target_names=np.array(["resistant", "sensitive"]),
        	feature_names=features
        	)
        f1.write("Dataset:\n%s\n\n" % dataset)

        # Defining linear regression parameters    
        if penalty == 'l1' or "L1":
            lin_reg = Lasso(max_iter=max_iter, tol=tol)        
        if penalty == 'l2' or "L2":
            lin_reg = Ridge(max_iter=max_iter, tol=tol)
        if penalty == 'elasticnet' or "L1+L2":
            lin_reg = ElasticNet(
                l1_ratio=l1_ratio, max_iter=max_iter, tol=tol
                )
        
        # Generate grid search classifier where parameters
        # (like regularization strength) are optimized by
        # cross-validated grid-search over a parameter grid.
        parameters = {'alpha': Samples.alphas}
        clf = GridSearchCV(lin_reg, parameters, cv=n_splits)

        # Fitting the linear regression model to dataset
        # (with or without considering the weights). Writing results
        # into corresponding files.
        if testset_size != 0.0:
            if penalty == 'l1' or "L1":
                (
                X_train, X_test, y_train, y_test, samples_train, samples_test
                ) = train_test_split(
                dataset.data, dataset.target, samples_in_analyze,
                test_size=testset_size, random_state=55
                )
                model = clf.fit(X_train, y_train)
            else:
                array_weights = np.array(
                    [process_input.samples[item].weight for item in samples_in_analyze]
                    )
                (
                X_train, X_test, sample_weights_train, sample_weights_test,
                y_train, y_test, samples_train, 
                samples_test
                ) = train_test_split(
                dataset.data, array_weights, dataset.target,
                samples_in_analyze, test_size=testset_size, random_state=55
                )
                model = clf.fit(
                    X_train, y_train,
                    sample_weight=sample_weights_train
                    )
            f1.write('Parameters:\n%s\n\n' % model)
            f1.write("Grid scores (R2 score) on development set: \n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(
            	    means, stds, clf.cv_results_['params']
            	    ):
                f1.write(
                	"%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params)
                	)
            f1.write("\nBest parameters found on development set: \n")
            for key, value in clf.best_params_.iteritems():
                f1.write(key + " : " + str(value) + "\n")      
            f1.write("\nModel predictions on test set:\nSample_ID " \
            	    "Acutal_phenotype Predicted_phenotype\n")
            for u in range(len(samples_test)):
                f1.write('%s %s %s\n' % (
                	samples_test[u], y_test[u], clf.predict(X_test)[u]
                	))
            train_y_prediction = clf.predict(X_train)
            f1.write('\nTraining set:\n')
            f1.write('Mean squared error: %s\n' % \
            	     mean_squared_error(y_train, train_y_prediction))
            f1.write("The coefficient of determination:"
                + " %s\n" % clf.score(X_train, y_train))
            f1.write("The Spearman correlation coefficient and p-value:" \
                " %s, %s \n" % stats.spearmanr(y_train, train_y_prediction))
            slope, intercept, r_value, pval_r, std_err = \
                stats.linregress(y_train, train_y_prediction)
            f1.write("The Pearson correlation coefficient and p-value: " \
                    " %s, %s \n" % (r_value, pval_r))
            f1.write("The plus/minus 1 dilution factor accuracy (for MICs):" \
                " %s \n\n" % metrics.within_1_tier_accuracy(
                    y_train, train_y_prediction
                    )
                )


            test_y_prediction = clf.predict(X_test)
            f1.write('Test set:\n')
            f1.write('Mean squared error: %s\n' 
                % mean_squared_error(y_test, test_y_prediction))
            f1.write('The coefficient of determination:'
                + ' %s\n' % clf.score(X_test, y_test)) 
            f1.write("The Spearman correlation coefficient and p-value:" \
                + " %s, %s \n" % stats.spearmanr(y_test, test_y_prediction))
            slope, intercept, r_value, pval_r, std_err = \
                stats.linregress(y_test, test_y_prediction)
            f1.write("The Pearson correlation coefficient and p-value: " \
                    " %s, %s \n" % (r_value, pval_r))
            f1.write("The plus/minus 1 dilution factor accuracy (for MICs):" \
                " %s \n\n" % metrics.within_1_tier_accuracy(
                    y_test, test_y_prediction
                    )
                )
        else:
            if penalty == 'l1' or "L1":
                model = clf.fit(dataset.data, dataset.target)
            else:
                array_weights = np.array(
                    [process_input.samples[item].weight for item in samples_in_analyze]
                    )
                model = clf.fit(
                    dataset.data, dataset.target, sample_weight=array_weights
                    )
            f1.write('Parameters:\n%s\n\n' % model)
            f1.write("Grid scores (R2 score) on development set:")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score'] 
            for mean, std, params in zip(
            	    means, stds, clf.cv_results_['params']
            	    ):
                f1.write("%0.3f (+/-%0.03f) for %r \n" % (
                	mean, std * 2, params
                	))
            f1.write("\nBest parameters found on development set: \n")
            for key, value in clf.best_params_.iteritems():
                f1.write(key + " : " + str(value) + "\n")
            y_prediction = clf.predict(dataset.data) 
            f1.write('\nMean squared error on the dataset: %s\n' % \
            	    mean_squared_error(dataset.target, y_prediction))
            f1.write("The coefficient of determination of the dataset: " \
            	    "%s\n" % clf.score(X_train, y_train))
            f1.write('The Spearman correlation coefficient and p-value of ' \
                'the dataset: %s, %s \n' % stats.spearmanr(
                    dataset.target, y_prediction
                    )
                )
            slope, intercept, r_value, pval_r, std_err = \
                stats.linregress(dataset.target, y_prediction)
            f1.write("The Pearson correlation coefficient and p-value: " \
                    " %s, %s \n" % (r_value, pval_r))
            f1.write("The plus/minus 1 dilution factor accuracy (for MICs):" \
                " %s \n\n" % metrics.within_1_tier_accuracy(
                    dataset.target, y_prediction
                    )
                )

        joblib.dump(model, model_filename)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        f2.write("K-mer\tcoef._in_lin_reg_model\tNo._of_samples_with_k-mer\
        	    \tSamples_with_k-mer\n")
        for x in range(len(clf.best_estimator_.coef_)):
            samples_with_kmer = [i for i,j in zip(
            	samples_in_analyze, kmers_presence_matrix[x]
            	) if j != 0]
            f2.write("%s\t%s\t%s\t| %s\n" % (
            	features[x], clf.best_estimator_.coef_[x],
            	len(samples_with_kmer), " ".join(samples_with_kmer)
            	))        
        f1.close()
        f2.close()

def logistic_regression(
	    kmer_lists_splitted,
	    kmers_passed_all_phenotypes, penalty, n_splits, testset_size,
	    l1_ratio, max_iter, tol
	    ):
    # Applies the logistic regression modeling on k-mers
    # that passed the filtering by p-value of statistical test. K-mers
    # presence/absence (0/1) in samples are used as independent
    # parameters, resistance value (0/1) is used as dependent 
    # parameter.
    names_of_samples = process_input.samples.keys()
    if len(Samples.phenotypes_to_analyse) > 1:
        sys.stderr.write("\nConducting the logistic regression analysis:\n")
    elif Samples.headerline:
        sys.stderr.write("\nConducting the logistic regression analysis of " 
            +  Samples.phenotypes[0] + " data...\n")
    else:
        sys.stderr.write("\nConducting the logistic regression analysis...\n")

    for j, k in enumerate(Samples.phenotypes_to_analyse):
        #Open files to write results of logistic regression
        phenotype = Samples.phenotypes[k]
        if Samples.headerline:
            f1 = open(
                "summary_of_log_reg_analysis_" + phenotype + ".txt", "w+"
                )
            f2 = open("k-mers_and_coefficients_in_log_reg_model_" 
                     + phenotype + ".txt", "w+")
            model_filename = "log_reg_model_" + phenotype + ".pkl"
            if len(Samples.phenotypes_to_analyse) > 1:
                sys.stderr.write("\tregression analysis of " 
                    +  phenotype + " data...\n")
        elif Samples.no_phenotypes > 1:
            f1 = open("summary_of_log_reg_analysis_" + phenotype + ".txt", "w+")
            f2 = open("k-mers_and_coefficients_in_log_reg_model_" 
                     + phenotype + ".txt", "w+")
            model_filename = "log_reg_model_" + phenotype + ".pkl"
            sys.stderr.write("\tregression analysis of phenotype " 
                +  phenotype + " data...\n")
        else:
            f1 = open("summary_of_log_reg_analysis.txt", "w+")
            f2 = open("k-mers_and_coefficients_in_log_reg_model.txt", "w+")
            model_filename = "log_reg_model.pkl"
        
        if len(kmers_passed_all_phenotypes[j]) == 0:
            f1.write("No k-mers passed the step of k-mer selection for " \
                "regression analysis.\n")
            continue

        # Generating a binary k-mer presence/absence matrix and a list
        # of k-mer names based on information in k-mer_matrix.txt
        matrix_and_features = map(
            list, zip(
                *process_input.pool.map(
                    partial(
                        get_kmer_presence_matrix,
                        set(kmers_passed_all_phenotypes[j])
                        ),
                    kmer_lists_splitted
                    )
                )
            )
        kmers_presence_matrix = [
            item for sublist in matrix_and_features[0] for item in sublist
            ]
        features = [
            item for sublist in matrix_and_features[1] for item in sublist
            ]
        Phenotypes = [item.phenotypes[k] for item in process_input.samples.values()]

        # Converting data into Python array formats suitable to use in
        # sklearn modeling. Also, deleting information associated with
        # stains missing the phenotype data
        features = np.array(features)
        Phenotypes = np.array(Phenotypes)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        samples_in_analyze = np.array(names_of_samples)
        to_del = []
        for i, item in enumerate(Phenotypes):
            if item == "NA":
                to_del.append(i)
        kmers_presence_matrix = np.delete(kmers_presence_matrix, to_del, 0)
        Phenotypes = map(int, np.delete(Phenotypes, to_del, 0))            
        samples_in_analyze = np.delete(samples_in_analyze, to_del, 0)

        #Insert data into logistic regression dataset  
        dataset = sklearn.datasets.base.Bunch(
        	data=kmers_presence_matrix, target=Phenotypes,
        	target_names=np.array(["resistant", "sensitive"]),
        	feature_names=features
        	) 
        f1.write("Dataset:\n%s\n\n" % dataset)

        #Defining logistic regression parameters
        if penalty == "L1" or "l1":
            log_reg = LogisticRegression(
                penalty='l1', solver='saga',
                max_iter=max_iter, tol=tol
                )        
        elif penalty == "L2" or "l2":
            log_reg = LogisticRegression(
                penalty='l2', solver='saga',
                max_iter=max_iter, tol=tol
                )
        elif penalty == "elasticnet" or "L1+L2":
            log_reg = SGDClassifier(
                penalty='elasticnet', l1_ratio=l1_ratio,
                max_iter=max_iter, tol=tol, loss='log'
                )
        

        # Generate grid search classifier where parameters
        # (like regularization strength) are optimized by
        # cross-validated grid-search over a parameter grid. 
        if penalty == "l1" or "l2":
            Cs = list(map(lambda x: 1/x, Samples.alphas))
            parameters = {'C':Cs}
        if penalty == "elasticnet":
            parameters = {'alpha': Samples.alphas}
        clf = GridSearchCV(log_reg, parameters, cv=n_splits)

        

        # Fitting the logistic regression model to dataset 
        # (with or without considering the weights). Writing logistic
        # regression results into corresponding files.
        if testset_size != 0.0:
            array_weights = np.array(
            	[process_input.samples[item].weight for item in samples_in_analyze]
            	)
            (
                X_train, X_test, sample_weights_train, sample_weights_test,
                y_train, y_test, samples_train, samples_test
                ) = train_test_split(
                dataset.data, array_weights, dataset.target,
                samples_in_analyze, test_size=testset_size,
                stratify=dataset.target, random_state=55
                )
            model = clf.fit(
            	X_train, y_train, sample_weight=sample_weights_train
            	)
            f1.write('Parameters:\n%s\n\n' % model)
            f1.write("Grid scores (mean accuracy) on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(
            	    means, stds, clf.cv_results_['params']
            	    ):
                f1.write("%0.3f (+/-%0.03f) for %r \n" % (
                	mean, std * 2, params
                	))
            f1.write("\nBest parameters found on development set: \n")
            for key, value in clf.best_params_.iteritems():
                f1.write(key + " : " + str(value) + "\n")
            f1.write("\n\nModel predictions on test set:\nSample_ID \
            	Acutal_phenotype Predicted_phenotype\n")
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            for u in range(len(samples_test)):
                f1.write('%s %s %s\n' % (
                	samples_test[u], y_test[u], y_test_pred[u]
                	))


            f1.write("\nTraining set: \n")
            f1.write("Mean accuracy: %s\n" % clf.score(X_train, y_train))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(y_train, y_train_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(lambda x: 1 if x == 0 else 0, y_train)), 
                        list(map(lambda x: 1 if x == 0 else 0, y_train_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(y_train, y_train_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_train, 
                    clf.predict_proba(X_train)[:,1]
                    )                        )
            f1.write("MCC: %s\n" % \
                matthews_corrcoef(y_train, y_train_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(y_train, y_train_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(y_train, y_train_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(y_train, y_train_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
                y_train, y_train_pred, 
                target_names=["sensitive", "resistant"]
                ))
            cm = confusion_matrix(y_train, y_train_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 


            f1.write("\nTest set: \n")
            f1.write('Mean accuracy: %s\n' % clf.score(X_test, y_test))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(y_test, y_test_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(lambda x: 1 if x == 0 else 0, y_test)), 
                        list(map(lambda x: 1 if x == 0 else 0, y_test_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(y_test, y_test_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_test, 
                    clf.predict_proba(X_test)[:,1])
                    )
            f1.write("MCC: %s\n" %\
                matthews_corrcoef(y_test, y_test_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(y_test, y_test_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(y_test, y_test_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(y_test, y_test_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
            	y_test, y_test_pred, 
            	target_names=["sensitive", "resistant"]
            	))
            cm = confusion_matrix(y_test, y_test_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 
        else:
            array_weights = np.array(
            	[process_input.samples[item].weight for item in samples_in_analyze])
            model = clf.fit(
            	dataset.data, dataset.target, 
            	fit_params={'sample_weight': array_weights}
            	)
            f1.write('Parameters:\n%s\n\n' % model)
            f1.write("Grid scores on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(
            	    means, stds, clf.cv_results_['params']
            	    ):
                f1.write("%0.3f (+/-%0.03f) for %r \n" % (
                	mean, std * 2, params
                	))
            f1.write("\nBest parameters found on development set: \n")
            for key, value in clf.best_params_.iteritems():
                f1.write(key + " : " + str(value) + "\n")
            y_pred = clf.predict(dataset.data)
            f1.write("\nMean accuracy on the dataset: %s\n" % clf.score(
            	dataset.data, dataset.target
            	))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(dataset.target, y_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(
                            lambda x: 1 if x == 0 else 0, dataset.target
                            )),
                        list(map(lambda x: 1 if x == 0 else 0, y_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(dataset.target, y_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_train, 
                    clf.predict_proba(X_train)[:,1])
                    )
            f1.write("MCC: %s\n" %\
                matthews_corrcoef(dataset.target, y_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(dataset.target, y_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(dataset.target, y_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(dataset.target, y_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
            	dataset.target, y_pred, 
            	target_names=["sensitive", "resistant"]
            	))
            cm = confusion_matrix(dataset.target, y_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 
        
        joblib.dump(model, model_filename)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        f2.write("K-mer\tcoef._in_log_reg_model\tNo._of_samples_with_k-mer\
        	    \tSamples_with_k-mer\n")
        for x in range(len(clf.best_estimator_.coef_[0])):
            samples_with_kmer = [i for i,j in zip(
            	samples_in_analyze, kmers_presence_matrix[x]
            	) if j != 0]
            f2.write("%s\t%s\t%s\t| %s\n" % (
            	features[x], clf.best_estimator_.coef_[0][x],
            	len(samples_with_kmer), " ".join(samples_with_kmer)
            	))
        f1.close()
        f2.close()

def support_vector_classifier(
        kmer_lists_splitted,
        kmers_passed_all_phenotypes, penalty, n_splits, testset_size,
        kernel, gammas, n_iter, max_iter, tol
        ):
    # Applies support vector machine modeling on k-mers
    # that passed the filtering by p-value of statistical test. K-mers
    # presence/absence (0/1) in samples are used as independent
    # parameters, resistance value (0/1) is used as dependent 
    # parameter.
    names_of_samples = process_input.samples.keys()
    if len(Samples.phenotypes_to_analyse) > 1:
        sys.stderr.write("\nConducting the SVM classifier analysis:\n")
    elif Samples.headerline:
        sys.stderr.write("\nConducting the SVM classifier analysis of " 
            +  Samples.phenotypes[0] + " data...\n")
    else:
        sys.stderr.write("\nConducting the SVM classifier analysis...\n")

    for j, k in enumerate(Samples.phenotypes_to_analyse):
        phenotype = Samples.phenotypes[k]
        #Open files to write results of logistic regression
        if Samples.headerline:
            f1 = open(
                "summary_of_SVM_analysis_" + phenotype + ".txt", "w+"
                )
            if kernel == "linear":
                f2 = open("k-mers_and_coefficients_in_SVM_model_" 
                         + phenotype + ".txt", "w+")
            model_filename = "SVM_model_" + phenotype + ".pkl"
            if len(Samples.phenotypes_to_analyse) > 1:
                sys.stderr.write("\tSVM analysis of " 
                    +  phenotype + " data...\n")
        elif Samples.no_phenotypes > 1:
            f1 = open("summary_of_SVM_analysis_" + phenotype + ".txt", "w+")
            if kernel == "linear":
                f2 = open("k-mers_and_coefficients_in_SVM_model_" 
                         + phenotype + ".txt", "w+")
            model_filename = "SVM_model_" + phenotype + ".pkl"
            sys.stderr.write("\tSVM analysis of phenotype " 
                +  phenotype + " data...\n")
        else:
            f1 = open("summary_of_SVM_analysis.txt", "w+")
            if kernel == "linear":
                f2 = open("k-mers_and_coefficients_in_SVM_model.txt", "w+")
            model_filename = "SVM_model.pkl"
        
        if len(kmers_passed_all_phenotypes[j]) == 0:
            f1.write("No k-mers passed the step of k-mer selection for \
                SVM analysis.\n")
            continue

        # Generating a binary k-mer presence/absence matrix and a list
        # of k-mer names based on information in k-mer_matrix.txt 
        matrix_and_features = map(
            list, zip(
                *process_input.pool.map(
                    partial(
                        get_kmer_presence_matrix,
                        set(kmers_passed_all_phenotypes[j])
                        ),
                    kmer_lists_splitted
                    )
                )
            )
        kmers_presence_matrix = [
            item for sublist in matrix_and_features[0] for item in sublist
            ]
        features = [
            item for sublist in matrix_and_features[1] for item in sublist
            ]
        Phenotypes = [item.phenotypes[k] for item in process_input.samples.values()]

        # Converting data into Python array formats suitable to use in
        # sklearn modeling. Also, deleting information associated with
        # stains missing the phenotype data
        features = np.array(features)
        Phenotypes = np.array(Phenotypes)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        samples_in_analyze = np.array(names_of_samples)
        to_del = []
        for i, item in enumerate(Phenotypes):
            if item == "NA":
                to_del.append(i)
        kmers_presence_matrix = np.delete(kmers_presence_matrix, to_del, 0)
        Phenotypes = map(int, np.delete(Phenotypes, to_del, 0))            
        samples_in_analyze = np.delete(samples_in_analyze, to_del, 0)

        #Insert data into logistic regression dataset  
        dataset = sklearn.datasets.base.Bunch(
            data=kmers_presence_matrix, target=Phenotypes,
            target_names=np.array(["resistant", "sensitive"]),
            feature_names=features
            ) 
        f1.write("Dataset:\n%s\n\n" % dataset)

        #Defining logistic regression parameters
        svc = SVC(kernel=kernel, probability=True, max_iter=max_iter, tol=tol)        
        

        # Generate grid search classifier where parameters
        # (like regularization strength) are optimized by
        # cross-validated grid-search over a parameter grid. 
        Cs = list(map(lambda x: 1/x, Samples.alphas))
        Gammas = list(map(lambda x: 1/x, Samples.alphas))
        if kernel == "linear":
            parameters = {'C':Cs}
            clf = GridSearchCV(svc, parameters, cv=n_splits)
        if kernel == "rbf":
            parameters = {'C':Cs, 'gamma':Gammas}
            clf = RandomizedSearchCV(svc, parameters, n_iter=n_iter, cv=n_splits)

        

        # Fitting the logistic regression model to dataset 
        # (with or without considering the weights). Writing logistic
        # regression results into corresponding files.
        if testset_size != 0.0:
            array_weights = np.array(
                [process_input.samples[item].weight for item in samples_in_analyze]
                )
            (
                X_train, X_test, sample_weights_train, sample_weights_test,
                y_train, y_test, samples_train, samples_test
                ) = train_test_split(
                dataset.data, array_weights, dataset.target,
                samples_in_analyze, test_size=testset_size,
                stratify=dataset.target, random_state=55
                )
            model = clf.fit(
                X_train, y_train, sample_weight=sample_weights_train
                )
            f1.write('Parameters:\n%s\n\n' % model)
            f1.write("Grid scores (mean accuracy) on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(
                    means, stds, clf.cv_results_['params']
                    ):
                f1.write("%0.3f (+/-%0.03f) for %r \n" % (
                    mean, std * 2, params
                    ))
            f1.write("\nBest parameters found on development set: \n")
            for key, value in clf.best_params_.iteritems():
                f1.write(key + " : " + str(value) + "\n")
            f1.write("\n\nModel predictions on test set:\nSample_ID \
                Acutal_phenotype Predicted_phenotype\n")
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            for u in range(len(samples_test)):
                f1.write('%s %s %s\n' % (
                    samples_test[u], y_test[u], y_test_pred[u]
                    ))


            f1.write("\nTraining set: \n")
            f1.write("Mean accuracy: %s\n" % clf.score(X_train, y_train))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(y_train, y_train_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(lambda x: 1 if x == 0 else 0, y_train)), 
                        list(map(lambda x: 1 if x == 0 else 0, y_train_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(y_train, y_train_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_train, 
                    clf.predict_proba(X_train)[:,1]
                    )                        )
            f1.write("MCC: %s\n" % \
                matthews_corrcoef(y_train, y_train_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(y_train, y_train_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(y_train, y_train_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(y_train, y_train_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
                y_train, y_train_pred, 
                target_names=["sensitive", "resistant"]
                ))
            cm = confusion_matrix(y_train, y_train_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 


            f1.write("\nTest set: \n")
            f1.write('Mean accuracy: %s\n' % clf.score(X_test, y_test))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(y_test, y_test_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(lambda x: 1 if x == 0 else 0, y_test)), 
                        list(map(lambda x: 1 if x == 0 else 0, y_test_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(y_test, y_test_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_test, 
                    clf.predict_proba(X_test)[:,1])
                    )
            f1.write("MCC: %s\n" %\
                matthews_corrcoef(y_test, y_test_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(y_test, y_test_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(y_test, y_test_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(y_test, y_test_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
                y_test, y_test_pred, 
                target_names=["sensitive", "resistant"]
                ))
            cm = confusion_matrix(y_test, y_test_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 
        else:
            array_weights = np.array(
                [process_input.samples[item].weight for item in samples_in_analyze])
            model = clf.fit(
                dataset.data, dataset.target, 
                fit_params={'sample_weight': array_weights}
                )
            f1.write('Parameters:\n%s\n\n' % model)
            f1.write("Grid scores on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(
                    means, stds, clf.cv_results_['params']
                    ):
                f1.write("%0.3f (+/-%0.03f) for %r \n" % (
                    mean, std * 2, params
                    ))
            f1.write("\nBest parameters found on development set: \n")
            for key, value in clf.best_params_.iteritems():
                f1.write(key + " : " + str(value) + "\n")
            y_pred = clf.predict(dataset.data)
            f1.write("\nMean accuracy on the dataset: %s\n" % clf.score(
                dataset.data, dataset.target
                ))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(dataset.target, y_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(
                            lambda x: 1 if x == 0 else 0, dataset.target
                            )),
                        list(map(lambda x: 1 if x == 0 else 0, y_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(dataset.target, y_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_train, 
                    clf.predict_proba(X_train)[:,1])
                    )
            f1.write("MCC: %s\n" %\
                matthews_corrcoef(dataset.target, y_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(dataset.target, y_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(dataset.target, y_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(dataset.target, y_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
                dataset.target, y_pred, 
                target_names=["sensitive", "resistant"]
                ))
            cm = confusion_matrix(dataset.target, y_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 
        
        joblib.dump(model, model_filename)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        if kernel == "linear":
            f2.write("K-mer\tcoef._in_log_reg_model\tNo._of_samples_with_k-mer\
                    \tSamples_with_k-mer\n")
            for x in range(len(clf.best_estimator_.coef_[0])):
                samples_with_kmer = [i for i,j in zip(
                    samples_in_analyze, kmers_presence_matrix[x]
                    ) if j != 0]
                f2.write("%s\t%s\t%s\t| %s\n" % (
                    features[x], clf.best_estimator_.coef_[0][x],
                    len(samples_with_kmer), " ".join(samples_with_kmer)
                    ))
        f1.close()
        f2.close()

def random_forest(
	    kmer_lists_splitted, kmers_passed_all_phenotypes,
        n_splits, testset_size
	    ):
    # Applies random forest modeling on k-mers
    # that passed the filtering by p-value of statistical test. K-mers
    # presence/absence (0/1) in samples are used as independent
    # parameters, resistance value (0/1) is used as dependent 
    # parameter.
    
    names_of_samples = samples.keys()
    if len(Samples.phenotypes_to_analyse) > 1:
        sys.stderr.write("\nConducting the random forest analysis:\n")
    elif Samples.headerline:
        sys.stderr.write("\nConducting the random forest analysis of " 
            +  Samples.phenotypes[0] + " data...\n")
    else:
        sys.stderr.write("\nConducting the random forest analysis...\n")

    for j, k in enumerate(Samples.phenotypes_to_analyse):
        #Open files to write results of logistic regression
        phenotype = Samples.phenotypes[k]
        if Samples.headerline:
            f1 = open(
                "summary_of_RF_analysis_" + phenotype + ".txt", "w+"
                )
            f2 = open("k-mers_and_coefficients_in_RF_model_" 
                     + phenotype + ".txt", "w+")
            model_filename = "RF_model_" + phenotype + ".pkl"
            if len(Samples.phenotypes_to_analyse) > 1:
                sys.stderr.write("\trandom forest analysis of " 
                    +  phenotype + " data...\n")
        elif Samples.no_phenotypes > 1:
            f1 = open("summary_of_RF_analysis_" + phenotype + ".txt", "w+")
            f2 = open("k-mers_and_coefficients_in_RF_model_" 
                     + phenotype + ".txt", "w+")
            model_filename = "RF_model_" + phenotype + ".pkl"
            sys.stderr.write("\trandom forest analysis of phenotype " 
                +  phenotype + " data...\n")
        else:
            f1 = open("summary_of_RF_analysis.txt", "w+")
            f2 = open("k-mers_and_coefficients_in_RF_model.txt", "w+")
            model_filename = "RF_model.pkl"
        
        if len(kmers_passed_all_phenotypes[j]) == 0:
            f1.write("No k-mers passed the step of k-mer selection for \
                random forest analysis.\n")
            continue

        # Generating a binary k-mer presence/absence matrix and a list
        # of k-mer names based on information in k-mer_matrix.txt
        matrix_and_features = map(
            list, zip(
                *process_input.pool.map(
                    partial(
                        get_kmer_presence_matrix,
                        set(kmers_passed_all_phenotypes[j])
                        ),
                    kmer_lists_splitted
                    )
                )
            )
        kmers_presence_matrix = [
            item for sublist in matrix_and_features[0] for item in sublist
            ]
        features = [
            item for sublist in matrix_and_features[1] for item in sublist
            ]
        Phenotypes = [item.phenotypes[k] for item in process_input.samples.values()]

        # Converting data into Python array formats suitable to use in
        # sklearn modeling. Also, deleting information associated with
        # stains missing the phenotype data
        features = np.array(features)
        Phenotypes = np.array(Phenotypes)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        samples_in_analyze = np.array(names_of_samples)
        to_del = []
        for i, item in enumerate(Phenotypes):
            if item == "NA":
                to_del.append(i)
        kmers_presence_matrix = np.delete(kmers_presence_matrix, to_del, 0)
        Phenotypes = map(int, np.delete(Phenotypes, to_del, 0))            
        samples_in_analyze = np.delete(samples_in_analyze, to_del, 0)

        #Insert data into logistic regression dataset  
        dataset = sklearn.datasets.base.Bunch(
        	data=kmers_presence_matrix, target=Phenotypes,
        	target_names=np.array(["resistant", "sensitive"]),
        	feature_names=features
        	) 
        f1.write("Dataset:\n%s\n\n" % dataset)

        #Defining logistic regression parameters
        clf = RandomForestClassifier(n_estimators=100) 


        # Fitting the random forest model to dataset 
        # (with or without considering the weights). Writing logistic
        # regression results into corresponding files.
        if testset_size != 0.0:
            array_weights = np.array(
            	[process_input.samples[item].weight for item in samples_in_analyze]
            	)
            (
                X_train, X_test, sample_weights_train, sample_weights_test,
                y_train, y_test, samples_train, samples_test
                ) = train_test_split(
                dataset.data, array_weights, dataset.target,
                samples_in_analyze, test_size=testset_size,
                stratify=dataset.target, random_state=55
                )
            model = clf.fit(
            	X_train, y_train, sample_weight=sample_weights_train
            	)
            f1.write("\n\nModel predictions on test set:\nSample_ID \
            	Acutal_phenotype Predicted_phenotype\n")
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            for u in range(len(samples_test)):
                f1.write('%s %s %s\n' % (
                	samples_test[u], y_test[u], y_test_pred[u]
                	))


            f1.write("\nTraining set: \n")
            f1.write("Mean accuracy: %s\n" % clf.score(X_train, y_train))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(y_train, y_train_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(lambda x: 1 if x == 0 else 0, y_train)), 
                        list(map(lambda x: 1 if x == 0 else 0, y_train_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(y_train, y_train_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_train, 
                    clf.predict_proba(X_train)[:,1]
                    )                        )
            f1.write("MCC: %s\n" % \
                matthews_corrcoef(y_train, y_train_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(y_train, y_train_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(y_train, y_train_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(y_train, y_train_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
                y_train, y_train_pred, 
                target_names=["sensitive", "resistant"]
                ))
            cm = confusion_matrix(y_train, y_train_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 


            f1.write("\nTest set: \n")
            f1.write('Mean accuracy: %s\n' % clf.score(X_test, y_test))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(y_test, y_test_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(lambda x: 1 if x == 0 else 0, y_test)), 
                        list(map(lambda x: 1 if x == 0 else 0, y_test_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(y_test, y_test_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_test, 
                    clf.predict_proba(X_test)[:,1])
                    )
            f1.write("MCC: %s\n" %\
                matthews_corrcoef(y_test, y_test_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(y_test, y_test_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(y_test, y_test_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(y_test, y_test_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
            	y_test, y_test_pred, 
            	target_names=["sensitive", "resistant"]
            	))
            cm = confusion_matrix(y_test, y_test_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 
        else:
            array_weights = np.array(
            	[process_input.samples[item].weight for item in samples_in_analyze])
            model = clf.fit(
            	dataset.data, dataset.target, 
            	fit_params={'sample_weight': array_weights}
            	)
            f1.write('Parameters:\n%s\n\n' % model)
            f1.write("Grid scores on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(
            	    means, stds, clf.cv_results_['params']
            	    ):
                f1.write("%0.3f (+/-%0.03f) for %r \n" % (
                	mean, std * 2, params
                	))
            f1.write("\nBest parameters found on development set: \n")
            for key, value in clf.best_params_.iteritems():
                f1.write(key + " : " + str(value) + "\n")
            y_pred = clf.predict(dataset.data)
            f1.write("\nMean accuracy on the dataset: %s\n" % clf.score(
            	dataset.data, dataset.target
            	))
            f1.write("Sensitivity: %s\n" % \
                    recall_score(dataset.target, y_pred))
            f1.write("Specificity: %s\n" % \
                    recall_score(
                        list(map(
                            lambda x: 1 if x == 0 else 0, dataset.target
                            )),
                        list(map(lambda x: 1 if x == 0 else 0, y_pred
                        ))))
            f1.write("AUC-ROC: %s\n" % \
                roc_auc_score(dataset.target, y_pred, average="micro"))
            f1.write("Average precision: %s\n" % \
                average_precision_score(
                    y_train, 
                    clf.predict_proba(X_train)[:,1])
                    )
            f1.write("MCC: %s\n" %\
                matthews_corrcoef(dataset.target, y_pred))
            f1.write("Cohen kappa: %s\n" %\
                cohen_kappa_score(dataset.target, y_pred))
            f1.write("Very major error rate: %s\n" %\
                metrics.VME(dataset.target, y_pred))
            f1.write("Major error rate: %s\n" %\
                metrics.ME(dataset.target, y_pred))
            f1.write('Classification report:\n %s\n' % classification_report(
            	dataset.target, y_pred, 
            	target_names=["sensitive", "resistant"]
            	))
            cm = confusion_matrix(dataset.target, y_pred)
            f1.write("Confusion matrix:\n")
            f1.write("Predicted\t0\t1:\n")
            f1.write("Actual\n")
            f1.write("0\t\t%s\t%s\n" % tuple(cm[0]))
            f1.write("1\t\t%s\t%s\n\n" % tuple(cm[1])) 
        
        joblib.dump(model, model_filename)
        kmers_presence_matrix = np.array(kmers_presence_matrix).transpose()
        f2.write("K-mer\tcoef._in_log_reg_model\tNo._of_samples_with_k-mer\
                \tSamples_with_k-mer\n")
        for x in range(len(clf.feature_importances_)):
            samples_with_kmer = [i for i,j in zip(
                samples_in_analyze, kmers_presence_matrix[x]
                ) if j != 0]
            f2.write("%s\t%s\t%s\t| %s\n" % (
                features[x], clf.feature_importances_[x],
                len(samples_with_kmer), " ".join(samples_with_kmer)
                ))
        f1.close()
        f2.close()


def ReverseComplement(kmer):
    # Returns the reverse complement of kmer
    seq_dict = {'A':'T','T':'A','G':'C','C':'G'}
    return("".join([seq_dict[base] for base in reversed(kmer)]))

def string_set(string_list):
    # Removes subsequences from kmer_list
    return set(i for i in string_list
               if not any(i in s for s in string_list if i != s))

def overlap(a, b, min_length=3):
    # Returns the overlap of kmer_a and kmer_b if overlap equals or 
    # exceeds the min_length. Otherwise returns 0.
    start = 0
    while True:
        start = a.find(b[:min_length], start)
        if start == -1:
            return 0
        if b.startswith(a[start:]):
            return len(a) - start
        start += 1

def pick_overlaps(reads, min_olap):
    # Takes kmer_list as an input. Generates pairwise permutations of 
    # the kmers in kmer list. Finds the overlap of each pair. Returns 
    # the lists of kmers and overlap lengths of the pairs which overlap
    # by min_olap or more nucleotides.
    reada, readb, olap_lens = [], [], []
    for a, b in permutations(reads, 2):
        olap_len = overlap(a, b, min_length=min_olap)
        if olap_len > 0:
            reada.append(a)
            readb.append(b)
            olap_lens.append(olap_len)
    return reada, readb, olap_lens

def kmer_assembler(kmer_list, min_olap=None):
    # Assembles the k-mers in kmer_list which overlap by at least 
    # min_olap nucleotides.

    kmer_length = len(kmer_list[0])
    if min_olap == None:
        min_olap = kmer_length-1
    assembled_kmers = []

    # Adding the reverse-complement of each k-mer
    kmer_list = kmer_list + map(ReverseComplement, kmer_list)

    # Find the overlaping k-mers
    kmers_a, kmers_b, olap_lens = pick_overlaps(kmer_list, min_olap)

    while olap_lens != []:
        set_a = set(kmers_a)
        set_b = set(kmers_b)

        # Picking out the assembled k-mers which have no sufficient
        # overlaps anymore.
        for item in kmer_list:
            if (item not in set_a and item not in set_b
                    and ReverseComplement(item) not in assembled_kmers):
                assembled_kmers.append(item)

        # Generating new kmer_list, where overlaping elements from previous
        # kmer_list are assembled.
        kmer_list = []
        for i, olap in enumerate(olap_lens):
            kmer_list.append(kmers_a[i] + kmers_b[i][olap:])

        # Removing substrings of other elements from kmer_list.
        kmer_list = list(string_set(kmer_list))

        # Find the overlaping elements in new generated kmer_list.
        kmers_a, kmers_b, olap_lens = pick_overlaps(kmer_list, min_olap)

    for item in kmer_list:
        # Picking out the assembled k-mers to assembled_kmers set.
        if (ReverseComplement(item) not in assembled_kmers):
            assembled_kmers.append(item)
    return(assembled_kmers)

def assembling(kmers_passed_all_phenotypes, phenotypes_to_analyze):
    # Assembles the input k-mers and writes assembled sequences
    # into "assembled_kmers.txt" file in FastA format.

    if len(phenotypes_to_analyze) > 1:
        sys.stderr.write(
            "Assembling the k-mers used in regression modeling of:\n"
            )
    elif Samples.headerline:
        sys.stderr.write("Assembling the k-mers used in modeling of " 
            +  Samples.phenotypes[0] + " data...\n")
    else:
        sys.stderr.write(
            "Assembling the k-mers used in modeling...\n"
            )

    for j, k in enumerate(phenotypes_to_analyze):
        phenotype = Samples.phenotypes[k]
        if len(kmers_passed_all_phenotypes[j]) == 0:
            f1.write("No k-mers passed the step of k-mer selection for \
                assembling.\n")
            continue
        #Open files to write the results of k-mer assembling
        if Samples.headerline:
            f1 = open("assembled_kmers_" + phenotype + ".fasta", "w+")
            if len(phenotypes_to_analyze) > 1:
                sys.stderr.write("\t" + phenotype + "...\n")
        elif Samples.no_phenotypes > 1:
            f1 = open("assembled_kmers_" + phenotype + ".fasta", "w+")
            sys.stderr.write("\tphenotype " + phenotype + "...\n")
        else:
            f1 = open("assembled_kmers.fasta", "w+")
        
        kmers_to_assemble = kmers_passed_all_phenotypes[j]
        assembled_kmers = sorted(
            kmer_assembler(kmers_to_assemble), key = len
            )[::-1]
        for i, item in enumerate(assembled_kmers):
            f1.write(">seq_" + str(i+1) + "_length_" 
                + str(len(item)) + "\n" + item + "\n")
    f1.close()

def modeling(args):
    # The main function of "phenotypeseeker modeling"
    process_input.get_input_data(args.inputfile, args.take_logs)
    process_input.process_input_args(
        args.alphas, args.alpha_min, args.alpha_max, args.n_alphas,
        args.gammas, args.gamma_min, args.gamma_max, args.n_gammas,
        args.min, args.max, args.mpheno, args.length, args.cutoff,
        args.num_threads 
        )
    process_input.get_multithreading_parameters()
    sys.stderr.write("Generating the k-mer lists for input samples:\n")
    process_input.pool.map(
        lambda x: x.get_kmer_lists(), process_input.samples.values()
        )
    sys.stderr.write("\nGenerating the k-mer feature vector.\n")
    Samples.get_feature_vector()
    sys.stderr.write("Mapping samples to the feature vector space:\n")
    stderr_print.currentSampleNum.value = 0
    process_input.pool.map(lambda x: x.map_samples(), process_input.samples.values())
    #call(["rm -r K-mer_lists/"], shell = True)
    if args.weights == "+":
        Samples.get_weights()
    kmers.test_kmers_association_with_phenotype()
    for key, value in process_input.phenotypes_to_analyse.iteritems():
        print(len(value.pvalues))
        '''
    kmers_passed_all_phenotypes = kmer_filtering_by_pvalue(
        args.pvalue, pvalues_all_phenotypes, args.n_kmers, 
        args.FDR, args.Bonferroni
        )

    if Samples.phenotype_scale == "continuous":
        linear_regression(
            vectors_as_multiple_input,
            kmers_passed_all_phenotypes, args.regularization, args.n_splits,
            args.testset_size,
            args.l1_ratio, args.max_iter,
            args.tol
            )
    elif Samples.phenotype_scale == "binary":
        if args.binary_classifier == "log":
            logistic_regression(
                vectors_as_multiple_input,
                kmers_passed_all_phenotypes, args.regularization, args.n_splits,
                args.testset_size,
                args.l1_ratio, args.max_iter, 
                args.tol
                )
        elif args.binary_classifier == "SVM":
            support_vector_classifier(
                vectors_as_multiple_input,
                kmers_passed_all_phenotypes, args.regularization, args.n_splits,
                args.testset_size,
                args.kernel, gammas, args.n_iter, 
                args.max_iter, args.tol
                )
        elif args.binary_classifier == "RF":
        	random_forest(
                vectors_as_multiple_input,
                kmers_passed_all_phenotypes, args.n_splits,
                args.testset_size,
                )

    if args.assembly == "+":
        assembling(kmers_passed_all_phenotypes, args.mpheno)
        '''
import sys
import pwseqdist as pw
from tcrdist.repertoire import TCRrep
import argparse
import pathlib
import pandas
import multiprocessing
import typing
import numpy as np
import pandas
import dill # extension of pickle, but also allows you to save interpretor sessions if needed

'''
DISTANCE METRICS DICTIONARY FOR ALPHA AND BETA CHAINS
metrics options:
    * [DEFAULT] pw.metrics.nb_vector_tcrdist = a Numba accelerated metric that matches the approach in Dash et al. (2018)
    * pw.metrics.nw_hamming_metric = a pairwise Needleman-Wunsch alignment followed by hamming distance
    * pw.metrics.nw_metric = a reciprocal Needleman-Wunsch alignment score based dissimilarity using BLOSUM62
'''
def generate_distance_metrics_dict(chain:str) -> tuple[dict, dict | None]:
    print("Generating distance parameters dictionary...", flush = True) # otherwise statement can wait in memory instead of printing to screen immediately

    metrics_alpha = {
        "cdr3_a_aa" : pw.metrics.nb_vector_tcrdist,
        "pmhc_a_aa" : pw.metrics.nb_vector_tcrdist,
        "cdr2_a_aa" : pw.metrics.nb_vector_tcrdist,
        "cdr1_a_aa" : pw.metrics.nb_vector_tcrdist}

    metrics_beta = {
        "cdr3_b_aa" : pw.metrics.nb_vector_tcrdist,
        "pmhc_b_aa" : pw.metrics.nb_vector_tcrdist,
        "cdr2_b_aa" : pw.metrics.nb_vector_tcrdist,
        "cdr1_b_aa" : pw.metrics.nb_vector_tcrdist}

    if chain == "alpha":
        return metrics_alpha, None
    elif chain == "beta":
        return metrics_beta, None

    return metrics_alpha, metrics_beta




'''
WEIGHTING DICTIONARY FOR ALPHA AND BETA CHAINS
weights is an integer of how much weight you want to put on the similarity.  Example if cdr3_a_aa is 3 and cdr1_a_aa is 1, then the CDR3 alpha amino acid
14 sequence is weighted 3 times higher than cdr1_a_aa.  There is 3x more importance placed on the biochemical and positional sequence similarity at the CDR3 alpha amino
15 acide sequence than the CDR1. Typically the CDR3 is given a higher weight because it is in direct contact with antigen peptide.

peptide–Major Histocompatibility Complex (pMHC)
pmhc is the pMHC-facing loop between CDR2 and CDR3, typically referred to as CDR2.5
this is the molecular structure that presents antigen fragements (peptides)
'''
def generate_weights_dict(chain:str) -> tuple[dict, dict | None]:
    print("Generating weights parameters dictionary...", flush = True) # otherwise statement can wait in memory instead of printing to screen immediately

    weights_alpha = {
        "cdr3_a_aa" : 3,
        "pmhc_a_aa" : 1,
        "cdr2_a_aa" : 1,
        "cdr1_a_aa" : 1}

    weights_beta = {
        "cdr3_b_aa" : 3,
        "pmhc_b_aa" : 1,
        "cdr2_b_aa" : 1,
        "cdr1_b_aa" : 1}

    if chain == "alpha":
        return weights_alpha, None
    elif chain == "beta":
        return weights_beta, None

    return weights_alpha, weights_beta




'''
DISTANCE METRIC TUNING PARAMETERS DICTIONARY FOR ALPHA AND BETA CHAINS
these are passed into the metrics calculation to fine tune the distance metrics and how penalities work for gaps, mismatched aa penalties, trimming, etc...
    * use_numba = must be set to True for all calculations with nb
    * ctrim = # aa to trim from c-terminal end of CDR (i.e. ctrim = 3 CASSQDFEQ-YF would consider only positions after "CAS" so SQDFEQ-YF)
    * ntrim = # aa to trim from n-terminal end of CDR (i.e. ntrim = 2 CASSQDFEQ-YF consider only positions before "YF", so CASSQDFEQ)
    * gap_penalty = the penalty accrued for gaps in the pairwise alignment
    * fixed_gappos = When sequences are of different length, there will be a gap position. If ‘fixed_gappos’ is False, then the metric inserts a single gap at an optimal position based on a BLOSUM62 scoring matrix.
                     This is recommended for the CDR3, but it is not necessary when the CDR1, 2, and 2.5 are already imgt_aligned and of a fixed length.
'''
def generate_metrics_tuning_params_dict(chain:str) -> tuple[dict, dict | None]:
    print("Generating metrics tuning parameters dictionary...", flush = True) # otherwise statement can wait in memory instead of printing to screen immediately

    metrics_kargs_alpha = {
        'cdr3_a_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 3,
            'ctrim' : 2,
            'fixed_gappos' : False},
        'pmhc_a_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 0,
            'ctrim' : 0,
            'fixed_gappos' : True},
        'cdr2_a_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 0,
            'ctrim': 0,
            'fixed_gappos' : True},
        'cdr1_a_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 0,
            'ctrim' : 0,
            'fixed_gappos' : True}
        }


    metrics_kargs_beta = {
        'cdr3_b_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 3,
            'ctrim' : 2,
            'fixed_gappos' : False},
        'pmhc_b_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 0,
            'ctrim' : 0,
            'fixed_gappos' : True},
        'cdr2_b_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 0,
            'ctrim': 0,
            'fixed_gappos' : True},
        'cdr1_b_aa' : {
            'use_numba' : True,
            'distance_matrix' : pw.matrices.tcr_nb_distance_matrix,
            'dist_weight' : 1,
            'gap_penalty' : 4,
            'ntrim' : 0,
            'ctrim' : 0,
            'fixed_gappos' : True}
        }

    if chain == "alpha":
        return metrics_kargs_alpha, None
    elif chain == "beta":
        return metrics_kargs_beta, None

    return metrics_kargs_alpha, metrics_kargs_beta

'''
summarize_neighborhoods_based_on_radii() output interpretation

Column                              Interpretation
-----------------------------------------------------------------------------------------------------------------
radius	                            Maximum TCRdist distance allowed
n_total_neighborhoods	            Number of focal clonotype neighborhoods
n_non_singleton_neighborhoods	    Number containing at least two clonotype rows
n_unique_neighborhoods	            Number of distinct neighborhood membership sets (includes singleton neighborhoods; if most clonotypes are isolated this may be close to equal to the total numberof clonotypes in the input dataset.
median_neighbors                    Typical number of neighboring TCRs; If 0, At least half of your clonotype rows have no other clonotype within that radius.
mean_neighbors	                    Average number of neighbors;
max_neighbors	                    Largest number of neighbors around one TCR
pct_with_neighbors                  Percentage of TCRs having at least one neighbor; Ex: If 20, then only 20% of clonotype rows have at least one neighbor
n_neighborhoods_ge3                 Number of focal TCRs with neighborhoods containing at least 3 TCRs, including themselves
n_neighborhoods_ge5                 Number of focal TCRs with neighborhoods containing at least 5 TCRs, including themselves
mean_subjects_per_neighborhood	    Average number of distinct subjects represented in each neighborhood
median_subjects_per_neighborhood	Median number of distinct subjects represented
n_neighborhoods_multiple_subjects	Number of neighborhoods containing TCRs from at least 2 subjects
n_neighborhoods_all_subjects	    Number of neighborhoods containing TCRs from every subject
high_only_neighborhoods	            Focal neighborhoods containing only high clonotypes, including singletons; note: high is assuming the name of a subject in your input data, could be anything listed in your input data
high_only_neighborhoods_ge3	        High-only focal neighborhoods containing at least 3 clonotype rows; note: high is assuming the name of a subject in your input data, could be anything listed in your input data
low_only_neighborhoods	            Focal neighborhoods containing only low clonotypes, including singletons; note: low is assuming the name of a subject in your input data, could be anything listed in your input data
low_only_neighborhoods_ge3	        Low-only focal neighborhoods containing at least 3 clonotype rows; note: low is assuming the name of a subject in your input data, could be anything listed in your input data

**These are neighborhood counts, not numbers of independent clusters. Neighborhoods can overlap substantially.**
'''
def summarize_neighborhoods_based_on_radii(distance_mtx:numpy.ndarray, radii:list, all_subjects:numpy.ndarray, save_prefix:pathlib.Path) -> pandas.DataFrame:
    print("Calculating basics summary statistics metrics on each radii listed...", flush = True)
    radius_results = [] # temp store results in preparation to convert to pandas df at end

    for radius in radii:

        neighbors_including_focal = distance_mtx <= radius

        # Number of neighbors, excluding the focal TCR
        n_neighbors = (distance_mtx <= radius).sum(axis=1) - 1

        # Number of distinct neighborhood membership sets
        n_unique_neighborhoods = len({tuple(np.flatnonzero(row)) for row in neighbors_including_focal})

        # Number of distinct subjects represented in each neighborhood
        # For example, suppose a neighborhood contains 10 TCRs: 6 from subject A, 3 from subject B, 1 from subject C
        # Then n_subjects_per_neighborhood = 3, regardless of how many clonotypes each subject contributes.
        # Important: These counts include the focal TCR itself, whereas your n_neighbors variable excludes it. This is intentional because neighborhood
        # membership typically includes the focal TCR.
        n_subjects_per_neighborhood = np.zeros(distance_mtx.shape[0], dtype=int) # initiate matrix of zeroes equal to the length of focal TCRs
        subject_presence = {}

        for subject in np.unique(all_subjects):
            subject_presence[subject] = (neighbors_including_focal[:, all_subjects == subject].any(axis=1))
            n_subjects_per_neighborhood += subject_presence[subject] # update matrix above after start counting membership to a neighborhood

        # append the summarized results to the results list
        result = {
            "radius": radius,
            "n_total_neighborhoods": distance_mtx.shape[0],
            "n_non_singleton_neighborhoods": np.sum(n_neighbors > 0),
            "n_unique_neighborhoods": n_unique_neighborhoods,
            "median_neighbors": np.median(n_neighbors),
            "mean_neighbors": np.mean(n_neighbors),
            "max_neighbors": np.max(n_neighbors),
            "pct_with_neighbors": 100 * np.mean(n_neighbors > 0),
            "n_neighborhoods_ge3": np.sum(n_neighbors >= 2),
            "n_neighborhoods_ge5": np.sum(n_neighbors >= 4),
            "mean_subjects_per_neighborhood": np.mean(n_subjects_per_neighborhood),
            "median_subjects_per_neighborhood": np.median(n_subjects_per_neighborhood),
            "n_neighborhoods_multiple_subjects": np.sum(n_subjects_per_neighborhood > 1),
            "n_neighborhoods_all_subjects": np.sum(n_subjects_per_neighborhood == len(np.unique(all_subjects)))
            }


        # Number of neighborhoods exclusive to each subject
        for subject in np.unique(all_subjects):

            result[f"{subject}_only_neighborhoods"] = np.sum(subject_presence[subject] &(n_subjects_per_neighborhood == 1))

             # Exclusive neighborhoods containing at least 3 clonotype rows
            result[f"{subject}_only_neighborhoods_ge3"] = np.sum(
                                                            subject_presence[subject] &
                                                            (n_subjects_per_neighborhood == 1) &
                                                            (n_neighbors >= 2))

        radius_results.append(result)

    # convert metrics of dict into pandas df for easy viewing/reading
    radius_summary = pandas.DataFrame(radius_results)


    print(radius_summary, flush = True)
    radius_summary.to_csv(f"{save_prefix}_radii_summary_statisitics.csv", index = False)
    return radius_summary


'''
nn_unique asks whether a TCR neighborhood contains a disproportionate number of distinct amino acid clonotypes from one sample.
nn_abundance asks whether a TCR neighborhood accounts for a disproportionate number of observed T cells from one sample, including clonal expansion.
enrichment, pvalues, FDR is based on a Fisher's Exact Test if 2x2 or Chi-squared must be selected if more than a 2x2 contignecy table since Fisher's a just a 2x2 - for example:
    val_[0-n] = group name
    ct_[0-n] = total clonotype rows that are in the val group witht he matching _[0-n]

    val_0 = (high, MEM+) - samples listed as high that are part of the neighborhood cluster (dist <= radii)
    ct_0 = 55 - total counts for (high, MEM+) (if abundance, sums the counts column which includes clonal expansions, if unique = this is literally just counting the number of clonotype rows, not taking into account clonal expansions, just unique occurences)
    val_1 = (high, MEM-) - samples listed as high that are NOT part the neighboorhood cluster (dist > radii)
    ct_1 = 306 - total counts for (high, MEM-)
    val_2 = (low, MEM+) - samples listed as low that are part of the neighborhood cluster (dist <= radii)
    ct_2 = 109 - total counts for (low, MEM+)
    val_3 = (low, MEM-) - samples listed as low that are NOT part of the neighboorhood cluster (dist > radii)
    ct_3 = 1030 - total counts for (low MEM-)

    This means there the odds ratio and pval with FDR is bassed on a 2x2 contigency table:
          MEM+    MEM-
    high  55      306
    low   109     1030

         odds that MEM+ was high (55/109)     0.5046
    OR = ---------------------------------- = ------ = 1.6978 (NOTE: OR = 1 means outcome is equally likely in each group, therefore there is no difference)
         odds that MEM- was high (306/1030)   0.2972


    interpretation: odds ratio (OR) = 1.6978 indicates that the odds of being in the high group are approximately 1.70 times greater for MEM+ than for MEM−.
    Equivalently, MEM+ is associated with 69.8% higher odds of being in the high group compared with MEM−. OR = 1.70 does not mean MEM+ causes a high classification. It indicates a positive association
    between MEM+ status and the high classification. You could equally state that the odds of being MEM+ are 1.70 times higher in the high group than in the low group. Both statements describe
    the same association.

    NOTE: ODDS OF BEING X != PROBABILTY OF BEING X!!
                            MEM+                    MEM-
    probability of high     55/(55+109) = 0.3353    306/(306+1030) = 0.2290
    odds of high            55/109 = 0.5046         306/(1030) = 0.2971

    interpretation: MEM+ samples have a 33.54% probability of being high, compared with 22.90% for MEM− samples.
    This corresponds to a 10.64 percentage-point higher probability, whereas the odds are approximately 70% higher.

    Another way to report all of the above: MEM+ samples exhibited 1.70-fold higher odds of being classified as high compared
    with MEM− samples (OR = 1.70). The proportion classified as high was 33.5% among MEM+ samples and 22.9% among MEM− samples.
'''

def differential_neighbors(clones_df:pandas.DataFrame, distance_mtx:numpy.ndarray, radii:list, test_method:str, save_prefix:pathlib.Path) -> None:
    print("Compariing differential neighbors by subject using Fisher's Exact test...", flush = True)

    clones_df["count_unique"]=1

    for radius in radii:
        # calculates based on clonotype rows (not by cell abundance - i.e. clonal expansion)
        nn_unique = neighborhood_diff(
                clone_df=clones_df,
                pwmat=distance_mtx,
                count_col="count_unique",
                x_cols=["subject"],
                test_method = test_method,
                knn_radius=radius)

        # uses the count columns based on abundance seen in data (if provided by user) i.e. includes all clonal expansions
        nn_abundance = neighborhood_diff(
                clone_df=clones_df,
                pwmat=distance_mtx,
                count_col="count",
                x_cols=["subject"],
                test_method = test_method,
                knn_radius=radius)

        nn_unique.to_csv(f"{save_prefix}_radii_{radius}_{test_method}_by_unique_clontypes.csv", index = False)
        nn_abundance.to_csv(f"{save_prefix}_radii_{radius}_{test_method}_by_cell_abundance.csv", index = False)


######################################################################
############################ DEFINE PARSER ###########################
######################################################################
parser = argparse.ArgumentParser(description="TCRdist3 workflow")
parser.add_argument("--inputCSV", required = True, help = "path to csv file that is formatted to minimum standards for TCRdist3 input csv", type = pathlib.Path)
#parser.add_argument("--countCol", default = "count", help = "The total number of occurences of the clonotype for the sample. Note, this mostly matters when comparing neighborhoods, does not affect distance calculations; if you set the count to all 1, each clonotypes will be treated equally in terms of comparing abundances in neighborhoods downstream.", type = str)
parser.add_argument("--chain", required = True, choices = ['alpha', 'beta', 'paired_ab'], help = "chains to base clustering upon or paired_ab if want both together")
parser.add_argument("--db_file", required = True, help = "TCRdist supplied or generated yourself from TCRdist helper script of database file that contains the database you want to use for alignment. Must end in .tsv for tab-delimited database files or .csv for comma-separated database files.", type = pathlib.Path)
parser.add_argument("--organism", required = True, help = "organism to use - this filters out the TCRdist database file to only the orgnaism listed in the file.", choices = ["human", "mouse"])
parser.add_argument("--cores", default = 1, type = int, help = "Number of CPU cores to use for parallel computing")
parser.add_argument("--save_file", required = True, type = pathlib.Path, help = "name to save TCRdist output - no suffix required, just prefix name, suffix of file types will automatically be appended.")
parser.add_argument("--test_radii", default = [8,12,16,20,24,32,36,40,48,50,60,65,75], type = int, nargs="+", help = "a list of radii distances to test to determine cluster membership. 8-16 = highly conserved, closely related TCRs 1-3 aa changes; 20-26 = intermediate sequence similarity; 40-50 = broad TCR neighborhoods, 60-75=permissive neighborhods, but still explored in tcrdist3 docs")
parser.add_argument("--test_method", default = "fishers", choices = ["fishers", "chi2", "chm"], help = "the statistical test to use when comparing differential neighbors for clonotypes.  fishers = Fishers exact test, chi2 = Chi-squared, chm = Cochran-Mantel-Haenszel test")
args = parser.parse_args()

# pre-flight checks
assert args.inputCSV.is_file(), f"File not found: {args.inputCSV}"  # confirm file exists and is file and not a folder
assert args.db_file.is_file(), f"File not found: {args.db_file}" # confirm file exists and is file and not a folder

######################################################################
############################ RUN TCRDIST3 ############################
######################################################################

# read in input csv file
if str(args.inputCSV).lower().endswith(".csv"):
    input_seqs_df = pandas.read_csv(args.inputCSV)
elif str(args.inputCSV).lower().endswith(".tsv"):
    input_seqs_df = pandas.read_csv(args.inputCSV, sep = "\t")
else:
    sys.exit("Input file does not end in .csv or .tsv.  Please rename file and be sure to convert file to either a csv or tsv.")

# generate parameters dict based on chain selection
dist1, dist2 = generate_distance_metrics_dict(chain = args.chain)
weight1, weight2 = generate_weights_dict(chain = args.chain)
tuning1, tuning2 = generate_metrics_tuning_params_dict(chain = args.chain)

# create TCRdist object
tcrdist_obj = TCRrep(
    cell_df=input_seqs_df,
    organism=args.organism,
    chains= ['alpha', 'beta'] if args.chain == 'paired_ab' else [args.chain],
    db_file=args.db_file,
    use_defaults=False,
    compute_distances=False,
    cpus = args.cores
)

# link parameter dictionaries to object
if args.chain == "alpha":
    tcrdist_obj.metrics_a = dist1
    tcrdist_obj.weights_a = weight1
    tcrdist_obj.kargs_a = tuning1
elif args.chain == "beta":
    tcrdist_obj.metrics_b = dist1
    tcrdist_obj.weights_b = weight1
    tcrdist_obj.kargs_b = tuning1
elif args.chain == "paired_ab":
    tcrdist_obj.metics_a = dist1
    tcrdist_obj.metrics_b = dist2
    tcrdist_obj.weights_a = weight1
    tcrdist_obj.weights_b = weight2
    tcrdist_obj.kargs_a = tuning1
    tcrdist_obj.kargs_b = tuning2


# Calculate distances using the linked parameter dictionaries
# note distances are not afffected by the counts column in the input data frame
tcrdist_obj.compute_distances()
dill.dump(tcrdist_obj, open(f"{args.save_file}.dill", mode='wb')) # save tcrdist object with computed distances and parameter dictionaries attached

if args.chain == "alpha":
    print("Extract radii distances for alpha chain only", flush = True)
    distance_matrix = tcrdist_obj.pw_alpha
elif args.chain == "beta":
    print("Extract radii distances for beta chain only", flush = True)
    distance_matrix = tcrdist_obj.pw_beta
elif args.chain == "paired_ab":
    print("Extract radii distances for alpha + beta chains calculated together", flush = True)
    distance_matrix = tcrdist_obj.pw_alpha + tcrdist_obj.pw_beta # this is how TCRdist handles running together; add the two distances together


# generate summary statistics to help make informed decision on radii to use
all_subjects = tcrdist_obj.clone_df["subject"].to_numpy() # generate the total unique sample names
summary_stats_df = summarize_neighborhoods_based_on_radii(distance_mtx = distance_matrix, radii = args.test_radii, all_subjects = all_subjects, save_prefix = args.save_file)


# differntial comparison neighbors
differential_neighbors(clones_df = tcrdist_obj.clone_df, distance_mtx = distance_matrix, radii = args.test_radii, test_method = args.test_method, save_prefix = args.save_file)

# save generated clones df
tcrdist_obj.clone_df.to_csv("{args.save_file}_tcrdist_clones_output.csv")

#tcrdist_obj = dill.load(open("yourfile.dill", mode='rb')) # example of how to load it in; saving this code for use later

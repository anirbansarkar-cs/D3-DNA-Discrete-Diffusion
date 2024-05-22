import numpy as np
import scipy
from scipy import linalg
import sklearn
import Bio 
from pymemsuite import fimo
from captum.attr import GradientShap
import logomaker
import tqdm


#############################################################################################
# Functional similarity: Conditional generation fidelity
#############################################################################################
'''
prerequisite: 
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A)) 
    - oracle (inference model that maps x to activities)

example:
    activity1 = oracle.predict(x_synthetic)
    activity2 = oracle.predict(x_test)
    mse = conditional_generation_fidelity(activity1, activity2)
'''

def conditional_generation_fidelity(activity1, activity2):
    return np.mean((activity1 - activity2)**2)


#############################################################################################
# Functional similarity: Frechet distance
#############################################################################################
'''
prerequisite: 
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A)) 
    - oracle_embedding_fun (function that acquires the penultimate embeddings)

example:
    embeddings1 = oracle_embedding_fun(x_synthetic)
    embeddings2 = oracle_embedding_fun(x_test)
    mu1, sigma1 = calculate_activation_statistics(embeddings1)
    mu2, sigma2 = calculate_activation_statistics(embeddings2)
    distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
'''

def calculate_activation_statistics(embeddings):
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    #Frechet distance: d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


#############################################################################################
# Functional similarity: Predictive distribution shift
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))
    - oracle model 

example:
    activity1 = oracle.predict(x_synthetic)
    activity2 = oracle.predict(x_test)
    mse = conditional_generation_fidelity(activity1, activity2)
'''

def conditional_generation_fidelity(activity1, activity2):
    return np.mean((activity1 - activity2)**2)



#############################################################################################
# Sequence similarity: Percent identity
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))

example:
    percent_identity = calculate_cross_sequence_identity_batch(x_synthetic, x_test, batch_size)
    max_percent_identity = np.max(percent_identity, axis=1)
    global_max_percent_identity = np.max(max_percent_identity)
'''

def calculate_cross_sequence_identity_batch(X_train, X_test, batch_size):
    num_train, seq_length, alphabet_size = X_train.shape    
    num_test = X_test.shape[0]
    
    # Reshape the matrices for dot product computation
    X_train = np.reshape(X_train, [-1, seq_length * alphabet_size])
    X_test = np.reshape(X_test, [-1, seq_length * alphabet_size])
    
    # Initialize the matrix to store the results
    seq_identity = np.zeros((num_train, num_test)).astype(np.int8)
    
    # Process the training data in batches
    for start_idx in tqdm(range(0, num_train, batch_size)):
        end_idx = min(start_idx + batch_size, num_train)
        
        # Compute the dot product for this batch
        batch_result = np.dot(X_train[start_idx:end_idx], X_test.T) 
        
        # Store the result in the corresponding slice of the output matrix
        seq_identity[start_idx:end_idx, :] = batch_result.astype(np.int8)
    
    return seq_identity


#############################################################################################
# Sequence similarity: k-mer spectrum shift
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))

example:
    kld, jsd = kmer_statistics(kmer_len, data1, data2)
'''

def kmer_statistics(kmer_length, data1, data2):

    #generate kmer distributions 
    dist1 = compute_kmer_spectra(data1, kmer_length)
    dist2 = compute_kmer_spectra(data2, kmer_length)

    #computer KLD
    kld = np.round(np.sum(scipy.special.kl_div(dist1, dist2)), 6)

    #computer jensen-shannon 
    jsd = np.round(np.sum(scipy.spatial.distance.jensenshannon(dist1, dist2)), 6)

    return kld, jsd

def compute_kmer_spectra(
    X,
    kmer_length=3,
    dna_dict = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
      }
    ):
    # convert one hot to A,C,G,T
    seq_list = []

    for index in tqdm(range(len(X))): #for loop is what actually converts a list of one-hot encoded sequences into ACGT

        seq = X[index]

        seq_list += ["".join([dna_dict[np.where(i)[0][0]] for i in seq])]

    obj = kmer_featurization(kmer_length)  # initialize a kmer_featurization object
    kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)

    kmer_permutations = ["".join(p) for p in product(["A", "C", "G", "T"], repeat=kmer_length)] #list of all kmer permutations, length specified by repeat=

    kmer_dict = {}
    for kmer in kmer_permutations:
        n = obj.kmer_numbering_for_one_kmer(kmer)
        kmer_dict[n] = kmer

    global_counts = np.sum(np.array(kmer_features), axis=0)

    # what to compute entropy against
    global_counts_normalized = global_counts / sum(global_counts) # this is the distribution of kmers in the testset
    # print(global_counts_normalized)
    return global_counts_normalized

class kmer_featurization:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'C', 'G', 'T']
        self.multiplyBy = 4 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
        self.n = 4**k # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.
        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        kmer_features = [] #a list containing the one-hot representation of kmers for each sequence in the list of sequences given
        for seq in seqs: #first obtain the one-hot representation of the kmers in a sequence
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature) #append this one-hot list into another list

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False): #
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.
        Args:
          seq:
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n) #array of zeroes the same length of all possible kmers

        for i in range(number_of_kmers): #for each kmer feature, turn the corresponding index in the list of all kmer features to 1
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer): #returns the corresponding index of a kmer in the larger list of all possible kmers?
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering

#############################################################################################
# Sequence similarity: Discriminatability
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_train (observed training sequences with shapes (N,L,A))
    - oracle model 

example:
    x_train = np.vstack([x_train, x_synthetic])
    y_train = np.vstack([np.ones((N,1)), np.zeros((N,1))])

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = train_val_test_split(
        x_train, y_train, val_size=0.1, test_size=0.2
    )

    # build model and train on dataset

    pred = model.predict(x_test)
    auroc = sklearn.metrics.roc_auc_score(y_test, pred) 
'''

def train_val_test_split(x_train, y_train, val_size=0.1, test_size=0.2):
    
    # Get the number of samples
    N = len(x_train)
    
    # Shuffle indices
    indices = np.random.permutation(N)
    
    # Calculate sizes of each split
    val_start = int(N * (1 - val_size - test_size))
    test_start = int(N * (1 - test_size))
    
    # Split indices
    train_idx = indices[:val_start]
    val_idx = indices[val_start:test_start]
    test_idx = indices[test_start:]
    
    # Split data
    x_train_split = x_train[train_idx]
    y_train_split = y_train[train_idx]
    
    x_val_split = x_train[val_idx]
    y_val_split = y_train[val_idx]
    
    x_test_split = x_train[test_idx]
    y_test_split = y_train[test_idx]
    
    return (x_train_split, y_train_split), (x_val_split, y_val_split), (x_test_split, y_test_split)


#############################################################################################
# Compositional similarity: Motif enrichment
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))
    - JASPAR_file (Datasbase for motif search)

example:
	x_synthetic = one_hot_to_seq(x_synthetic)
	x_test = one_hot_to_seq(x_test)
	create_fasta_file(x_synthetic,'sythetic_seq.txt')
	create_fasta_file(x_test,'test_seq.txt')
	motif_count = motif_count('test_seq.txt', JASPAR_file)
	motif_count_2 = motif_count('synthetic_seq.txt', JASPAR_file)
	pr = enrich_pr(motif_count,motif_count_2)
'''

def one_hot_to_seq(
    X,
    dna_dict = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
      }
    ):
    # convert one hot to A,C,G,T
    seq_list = []

    for index in tqdm(range(len(X))): #for loop is what actually converts a list of one-hot encoded sequences into ACGT

        seq = X[index]

        seq_list += ["".join([dna_dict[np.where(i)[0][0]] for i in seq])]

    return seq_list

def create_fasta_file(sequence_list, path):
    '''
    sequence_list is the input sequences to put into the fasta file
    path is the output filepath
    '''
    output_path = path
    output_file = open(output_path, 'w')
    for i in range(len(sequence_list)):
        identifier_line = '>Seq' + str(i) + '\n'
        output_file.write(identifier_line)
        sequence_line = sequence_list[i]
        output_file.write(sequence_line + '\n')

    output_file.close()
 
def motif_count(path, path_to_database):
    '''
    path is the filepath to the list of sequences in fasta format

    returns a dictionary containing the motif counts for all the sequences
    '''
    motifs, motif_file = load_jaspar_database(path_to_database)
    motif_ids = []
    occurrence = []

    sequences = [
        Sequence(str(record.seq), name=record.id.encode())
        for record in Bio.SeqIO.parse(path, "fasta")
        ]

    for motif in (motifs):
        pattern = fimo.score_motif(motif, sequences, motif_file.background)
        motif_ids.append(motif.accession.decode())
        occurrence.append(len(pattern.matched_elements))
    
    motif_counts = dict(zip(motif_ids,occurrence))

    return motif_counts

def enrich_pr(count_1,count_2):
	c_1 = list(count_1.values())
	c_2 = list(count_2.values())

	return scipy.stats.pearsonr(c_1,c_2)


#############################################################################################
# Compositional similarity: Motif co-occurrence
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))
    - JASPAR_file (Datasbase for motif search)

example:

	motif_matrix_test = FIMO_scanning.find_motifs_per_sequence(x_test, 'test_seq.txt', JASPAR_path)
    motif_matrix_synthetic = FIMO_scanning.find_motifs_per_sequence(x_synthetic, 'synthetic_seq.txt', JASPAR_path)
    C = np.cov(motif_matrix_test)
    C2 = np.cov(motif_matrix_synthetic)
    distance = frobenius_norm(C, C2)
'''

def covariance_matrix(x):
    return np.cov(x)


def frobenius_norm(cov, cov2):
    return np.sqrt(np.sum((cov - cov2)**2))


#############################################################################################
# Compositional similarity: Attribution maps
#############################################################################################
'''
prerequisite:
    - x_seq (generated or observed seqeunces with shapes (N,L,A)) 
    - oracle model 

example:
    shap_score = gradient_shap(x_seq, oracle, task_idx)
    plot_attribution_map(x_seq, shap_score)
'''
def gradient_shap(x_seq, model, class_index=0, trim_end=None):

	x_seq = np.swapaxes(x_seq,1,2)
	N,A,L = x_seq.shape
	score_cache = []

    for i,x in enumerate(x_seq):
        # process sequences so that they are right shape (based on insertions)
        x = np.expand_dims(x, axis=0)
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        x_tensor = model._pad_end(x_tensor)
        x = x_tensor.detach().numpy()

        # random background
        num_background = 1000
        null_index = np.random.randint(0,3, size=(num_background,L))
        x_null = np.zeros((num_background,A,L))
        for n in range(num_background):
            for l in range(L):
               x_null[n,null_index[n,l],l] = 1.0
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32)
        x_null_tensor = model._pad_end(x_null_tensor)

        # calculate gradient shap
        gradient_shap = GradientShap(model)
        grad = gradient_shap.attribute(x_tensor,
                                      n_samples=100,
                                      stdevs=0.1,
                                      baselines=x_null_tensor,
                                      target=class_index)
        grad = grad.data.cpu().numpy()

        # process gradients with gradient correction (Majdandzic et al. 2022)
        grad -= np.mean(grad, axis=1, keepdims=True)
       
        score_cache.append(np.squeeze(grad))

    score_cache = np.array(score_cache)        
    if len(score_cache.shape)<3:
        score_cache=np.expand_dims(score_cache,axis=0)
    if trim_end:
    	score_cache = score_cache[:,:,:-trim_end]
    
    return np.swapaxes(score_cache,1,2)

def plot_attribution_map(x_seq, shap_score, alphabet='ACGT', figsize=(20,1)):
	
	num_plot = len(x_seq)
	fig = plt.figure(figsize=(20,2*num_plot))
	
	i = 0
	for (x,grad) in zip(x_seq,shap_score):
		x_index = np.argmax(np.squeeze(x), axis=1)
		grad = np.squeeze(grad)
		L, A = grad.shape

		seq = ''
		saliency = np.zeros((L))
		for i in range(L):
			seq += alphabet[x_index[i]]
			saliency[i] = grad[i,x_index[i]]
		# create saliency matrix
		saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)

		ax = plt.subplot(num_plot,1,i+1)
		i+=1
		logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
			
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('none')
		ax.xaxis.set_ticks_position('none')
		plt.xticks([])
		plt.yticks([])


#############################################################################################
# Compositional similarity: Attribution consistency
#############################################################################################
'''
prerequisite:
    - x_seq (generated or observed seqeunces with shapes (N,L,A)) 
    - oracle model 

example:
    shap_score = gradient_shap(x_seq, oracle, task_idx)
    attributino_map = process_attribution_map(shap_score)
    mask = unit_mask(x_seq)
    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map], x_seq, mask, radius_count_cutoff)
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
	entropic_information = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
'''

def process_attribution_map(saliency_map_raw):
    saliency_map_raw = saliency_map_raw - np.mean(saliency_map_raw, axis=-1, keepdims=True) # gradient correction
    saliency_map_raw = saliency_map_raw / np.sum(np.sqrt(np.sum(np.square(saliency_map_raw), axis=-1, keepdims=True)), axis=-2, keepdims=True) #normalize
    saliency_map_raw_rolled = np.roll(saliency_map_raw, -1, axis=-2)
    saliency_map_raw_rolled_twice = np.roll(saliency_map_raw, -2, axis=-2)
    saliency_map_raw_rolled_triple = np.roll(saliency_map_raw, -3, axis=-2)
    saliency_map_raw_rolled_4 = np.roll(saliency_map_raw, -4, axis=-2)
    saliency_map_raw_rolled_5 = np.roll(saliency_map_raw, -5, axis=-2)
    saliency_map_raw_rolled_6 = np.roll(saliency_map_raw, -6, axis=-2)
    # Define k-window here, include k terms below (here k = 3)
    saliency_special = saliency_map_raw + saliency_map_raw_rolled + saliency_map_raw_rolled_twice #+ saliency_map_raw_rolled_triple # + saliency_map_raw_rolled_4 + saliency_map_raw_rolled_5 #This line is optional.
    saliency_special = ortonormal_coordinates(saliency_special) #Down to 3D, since data lives on the plane.
    return saliency_special

def unit_mask(x_seq):
	return np.sum(np.ones(x_seq.shape),axis=-1) / 4

def spherical_coordinates_process_2_trad(saliency_map_raw_s, X, mask, radius_count_cutoff=0.04):
	global N_EXP
    N_EXP = len(saliency_map_raw_s)
    radius_count=int(radius_count_cutoff * np.prod(X.shape)/4)
    cutoff=[]
    x_s, y_s, z_s, r_s, phi_1_s, phi_2_s = [], [], [], [], [], []
    for s in range (0, N_EXP):
        saliency_map_raw = saliency_map_raw_s[s]
        xxx_motif=saliency_map_raw[:,:,0]
        yyy_motif=(saliency_map_raw[:,:,1])
        zzz_motif=(saliency_map_raw[:,:,2])
        xxx_motif_pattern=saliency_map_raw[:,:,0]*mask
        yyy_motif_pattern=(saliency_map_raw[:,:,1])*mask
        zzz_motif_pattern=(saliency_map_raw[:,:,2])*mask
        r=np.sqrt(xxx_motif*xxx_motif+yyy_motif*yyy_motif+zzz_motif*zzz_motif)
        resh = X.shape[0] * X.shape[1]
        x=np.array(xxx_motif_pattern.reshape(resh,))
        y=np.array(yyy_motif_pattern.reshape(resh,))
        z=np.array(zzz_motif_pattern.reshape(resh,))
        r=np.array(r.reshape(resh,))
        #Take care of any NANs.
        x=np.nan_to_num(x)
        y=np.nan_to_num(y)
        z=np.nan_to_num(z)
        r=np.nan_to_num(r)
        cutoff.append( np.sort(r)[-radius_count] )
        R_cuttof_index = np.sqrt(x*x+y*y+z*z) > cutoff[s]
        #Cut off
        x=x[R_cuttof_index]
        y=y[R_cuttof_index]
        z=z[R_cuttof_index]
        r=np.array(r[R_cuttof_index])
        x_s.append(x)
        y_s.append(y)
        z_s.append(z)
        r_s.append(r)
        #rotate axis
        x__ = np.array(y)
        y__ = np.array(z)
        z__ = np.array(x)
        x = x__
        y = y__
        z = z__
        #"phi"
        phi_1 = np.arctan(y/x) #default
        phi_1 = np.where((x<0) & (y>=0), np.arctan(y/x) + PI, phi_1)   #overwrite
        phi_1 = np.where((x<0) & (y<0), np.arctan(y/x) - PI, phi_1)   #overwrite
        phi_1 = np.where (x==0, PI/2, phi_1) #overwrite
        #Renormalize temorarily to have both angles in [0,PI]:
        phi_1 = phi_1/2 + PI/2
        #"theta"
        phi_2=np.arccos(z/r)
        #back to list
        phi_1 = list(phi_1)
        phi_2 = list(phi_2)
        phi_1_s.append(phi_1)
        phi_2_s.append(phi_2)
    #print(cutoff)
    return phi_1_s, phi_2_s, r_s

def initialize_integration_2(box_length):
	LIM = 3.1416
    global volume_border_correction
    box_volume = box_length*box_length
    n_bins = int(LIM/box_length)
    volume_border_correction =  (LIM/box_length/n_bins)*(LIM/box_length/n_bins)
    #print('volume_border_correction = ', volume_border_correction)
    n_bins_half = int(n_bins/2)
    return LIM, box_length, box_volume, n_bins, n_bins_half

def calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, box_length, box_volume, prior_range):
    global Empirical_box_pdf_s
    global Empirical_box_count_s
    global Empirical_box_count_plain_s
    Empirical_box_pdf_s=[]
    Empirical_box_count_s = []
    Empirical_box_count_plain_s = []
    prior_correction_s = []
    Spherical_box_prior_pdf_s=[]
    for s in range (0,N_EXP):
        #print(s)
        Empirical_box_pdf_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[0])
        Empirical_box_count_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[1])
        Empirical_box_count_plain_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[2])
    Entropic_information = []
    for s in range (0,N_EXP):
        Entropic_information.append ( KL_divergence_2 (Empirical_box_pdf_s[s], Empirical_box_count_s[s], Empirical_box_count_plain_s[s], n_bins, box_volume, prior_range)  )
    return list(Entropic_information)

def KL_divergence_2(Empirical_box_pdf, Empirical_box_count, Empirical_box_count_plain, n_bins, box_volume, prior_range):  #, correction2)
    # p= empirical distribution, q=prior spherical distribution
    # Notice that the prior distribution is never 0! So it is safe to divide by q.
    # L'Hospital rule provides that p*log(p) --> 0 when p->0. When we encounter p=0, we would just set the contribution of that term to 0, i.e. ignore it in the sum.
    Relative_entropy = 0
    PI = 3.1416
    for i in range (1, n_bins-1):
        for j in range(1,n_bins-1):
            if (Empirical_box_pdf[i,j] > 0  ):
                phi_1 = i/n_bins*PI
                phi_2 = j/n_bins*PI
                correction3 = 0
                prior_counter = 0
                prior=0
                for ii in range(-prior_range,prior_range):
                    for jj in range(-prior_range,prior_range):
                        if(i+ii>0 and i+ii<n_bins and j+jj>0 and j+jj<n_bins):
                            prior+=Empirical_box_pdf[i+ii,j+jj]
                            prior_counter+=1
                prior=prior/prior_counter
                if(prior>0) : KL_divergence_contribution = Empirical_box_pdf[i,j] * np.log (Empirical_box_pdf[i,j]  /  prior )
                if(np.sin(phi_1)>0 and prior>0 ): Relative_entropy+=KL_divergence_contribution  #and Empirical_box_count_plain[i,j]>1
    Relative_entropy = Relative_entropy * box_volume #(volume differential in the "integral")
    return np.round(Relative_entropy,3)

def Empiciral_box_pdf_func_2 (phi_1, phi_2, r_s, n_bins, box_length, box_volume):
    N_points = len(phi_1) #Number of points
    Empirical_box_count = np.zeros((n_bins, n_bins))
    Empirical_box_count_plain = np.zeros((n_bins, n_bins))
    #Now populate the box. Go over every single point.
    for i in range (0, N_points):
        # k, l are box numbers of the (phi_1, phi_2) point
        k=np.minimum(int(phi_1[i]/box_length), n_bins-1)
        l=np.minimum(int(phi_2[i]/box_length), n_bins-1)
        #Increment count in (k,l,m) box:
        Empirical_box_count[k,l]+=1*r_s[i]*r_s[i]
        Empirical_box_count_plain[k,l]+=1
    #To get the probability distribution, divide the Empirical_box_count by the total number of points.
    Empirical_box_pdf = Empirical_box_count / N_points / box_volume
    #Check that it integrates to around 1:
    #print('Integral of the empirical_box_pdf (before first renormalization) = ' , np.sum(Empirical_box_pdf*box_volume), '(should be 1.0 if OK) \n')
    correction = 1 / np.sum(Empirical_box_pdf*box_volume)
    #Another, optional correction 
    count_empty_boxes = 0
    count_single_points = 0
    for k in range (1, n_bins-1):
        for l in range(1,n_bins-1):
            if(Empirical_box_count[k,l] ==1):
                count_empty_boxes+=1
                count_single_points+=1
    return Empirical_box_pdf * correction * 1 , Empirical_box_count *correction , Empirical_box_count_plain #, correction2


#############################################################################################
# Informativeness: Added information
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - y_synthetic (activities for sequences)
    - x_train (observed sequences with shapes (N,L,A))
    - y_train (activities for sequences)
    - x_valid (observed sequences with shapes (N,L,A))
    - y_valid (activities for sequences)
    - x_test (observed sequences with shapes (N,L,A))
    - y_test (activities for sequences)

example:
    downsample = 0.25
    N = x_train.shape[0]
    num_downsample = int(N*downsample)
    x_train = np.vstack([x_train[:num_downsample], x_synthetic])
    y_train = np.vstack([y_train[:num_downsample], y_synthetic])

    # train model using x_train, y_train

    # evaluate model on x_test, y_test
'''


#############################################################################################
# Conditional sampling diversity: sequence diversity
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 

example:
    max_percent_identity = sequence diversity(x_synthetic)
'''

def sequence diversity(x, batch_size):
    percent_identity = calculate_cross_sequence_identity_batch(x, x, batch_size)
    val = []
    for i in range(len(percent_identity)):
        sort = np.sort(percent_identity[i])[::-1] 
        val.append(sort[1]) # <-- take second highest percent identity due to match w/ self
    return np.array(val)



#############################################################################################
# Conditional sampling diversity: mechanistic diversity
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - y_test (observed functional activity used for conditional generation)
    - oracle model

example:
    shap_scores = gradient_shap(x_seq, oracle, task_idx)
    max_self_similarity = mechanistic_diversity(shap_scores)
    
'''

def mechanistic_diversity(attr_scores):
    N, L, A = attr_scores.shape    
    
    # Reshape the matrices for dot product computation
    attr_scores = np.reshape(attr_scores, [-1, L * A])
    
    # Initialize the matrix to store the results
    max_similarity = []    
    # Process the training data in batches
    for i in range(N):
        val = np.dot(np.expand_dims(attr_scores[i,:], axis=0), attr_scores.T) 
        max_similarity.append(np.sort(val)[::-1][1])
    
    return np.array(max_similarity)






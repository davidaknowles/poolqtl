import torch
import sys
import pyro

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

import numpy as np

from pooledQTL import deconvolve, beta_model, gaussian_model, io_utils, asb_data

from importlib import reload

from pathlib import Path

fdr_threshold = 0.05

results_dir = Path("/gpfs/commons/home/daknowles/pooledRBPs/results/")

# load input RNA allelic counts
input_files = [ "/gpfs/commons/home/mschertzer/asb_model/all-input-rep1_allelic.out", # this is bio rep 1 (two sequencing runs combined)
               "/gpfs/commons/home/mschertzer/pilot_pool/allelic/technical_rep1/input-rep2_allelic.out"] # bio rep 2, low sequencing depth

input_counts = [ pd.read_csv(f, sep = "\t", usecols = range(8), index_col = False) for f in input_files ]

# observed genotype data (after running StrandScript)
geno = io_utils.loadGenotypes('/gpfs/commons/home/phalmos/genotypes/CIRMlines_flipped.vcf', 
                     maxlines = None, 
                     posterior = False).rename(columns = {"SNP" : "variantID"})

# perform denconvolution
w = [ deconvolve.deconvolve(geno, inp) for inp in input_counts ]

props = pd.DataFrame(w).transpose()
props.columns = ["rep1","rep2"]
props["line"] = geno.columns[5:16]
props.iloc[:,[2,0,1]].to_csv(results_dir / "deconv.tsv", index = False, sep = "\t")

# load sanger (and cache) imputed data
sanger_feather = Path("/gpfs/commons/home/daknowles/pooledRBPs/genotypes/sanger.feather")
if sanger_feather.is_file(): 
    sanger = pd.read_feather(sanger_feather)
    del sanger["index"]
else: # this is pretty slow to read, so cache to feather
    sanger = io_utils.loadGenotypes("/gpfs/commons/home/daknowles/pooledRBPs/genotypes/sanger.vcf.gz", 
                           maxlines = None, 
                           posterior = True,
                           posterior_index = 2,
                          print_every = 10000) 
    sanger.reset_index().to_feather(sanger_feather)

cirm_lines = sanger.columns[5:16]
sanger_merge = geno.rename(columns = {"variantID" : "SNP"}).merge(sanger, on = ["SNP","refAllele","altAllele"], suffixes = ("_geno","_imp"))
geno_geno = sanger_merge.loc[:,cirm_lines.astype(str) + "_geno"].to_numpy()
geno_imp = sanger_merge.loc[:,cirm_lines.astype(str) + "_imp"].to_numpy()

### check imputation looks sensible on genotyped variants 
sns.kdeplot(geno_imp[geno_geno==0.], bw = 0.03, label = "Genotype 0")
sns.kdeplot(geno_imp[geno_geno==.5], bw = 0.03, label = "Genotype 1")
sns.kdeplot(geno_imp[geno_geno==1.], bw = 0.03, label = "Genotype 2")
plt.xlabel("Imputed dosage")
plt.ylabel("Density")
plt.legend()
plt.show() # nice! 

IP_files = [ "/gpfs/commons/home/mschertzer/asb_model/all_hnrnpk-rep1_allelic.out", 
             "/gpfs/commons/home/mschertzer/pilot_pool/allelic/technical_rep1/hnrnpk-rep2_allelic.out" ]
IP_counts = [ pd.read_csv(f, sep = "\t", usecols = range(8), index_col = False) for f in IP_files ]

merged,dat_sub = deconvolve.merge_geno_and_counts(sanger, input_counts[0], IP_counts[0], w[0], plot = True)
merged_2,dat_sub_2 = deconvolve.merge_geno_and_counts(sanger, input_counts[1], IP_counts[1], w[1], plot = True,
        input_total_min = 10, allele_count_min = 2, ip_total_min = 10) # more lenient thresholds for lower seq depth

dat_sub["input_ratio"] = dat_sub.altCount_input / dat_sub.totalCount_input
dat_sub["IP_ratio"] = dat_sub.altCount_IP / dat_sub.totalCount_IP

device = "cpu"
data = asb_data.RelativeASBdata.from_pandas(dat_sub, device = device)

for use_structured_guide in (False,True):

    print("Structured guide" if use_structured_guide else "Mean field guide")
    
    losses, model, guide, stats, samples = beta_model.fit(data, use_structured_guide = use_structured_guide)

    plt.plot(losses)
    plt.show()
    losses[-1] # 133094 compared to 117421 for structured SVI (so guess latter is better)

    p = (samples["input_ratio"] > samples["IP_ratio"]).float().mean(0).squeeze().numpy()

    # proportion significant
    dat_sub["q"] = np.minimum(p,1.-p)
    
    effect_size = torch.logit(samples["IP_ratio"]) - torch.logit(samples["input_ratio"])
    dat_sub["effect_mean"] = effect_size.mean(0).squeeze().numpy()
    dat_sub["effect_std"] = effect_size.std(0).squeeze().numpy()

    dat_sub.drop(columns = ["input_ratio", "IP_ratio"]).to_csv(results_dir / "beta" + ("_struct" if use_structured_guide else "") + "_results.tsv.gz", index = False, sep = "\t")

    plt.scatter(dat_sub.input_ratio, dat_sub.IP_ratio,alpha=0.1, color="gray")
    dat_ss = dat_sub[dat_sub.q < fdr_threshold]
    plt.scatter(dat_ss.input_ratio, dat_ss.IP_ratio,alpha=0.03, color = "red")
    plt.xlabel("Input proportion alt"); plt.ylabel("IP proportion alt")
    plt.title('%i (%.1f%%) significant %.0f%% FDR' % ((dat_sub["q"] < 0.1).sum(), 100. * (dat_sub["q"] < 0.1).mean(), fdr_threshold*100))
    plt.show()
    
    print(dat_sub.sort_values("q").head(20))

###################### normal model without replicates #############################

losses, model, guide, stats, samples = gaussian_model.fit(data, use_structured_guide = True)

{ k:v["mean"].item() for k,v in stats.items() if v["mean"].numel() ==1 }

p = (samples['asb'] > 0.).float().mean(0).squeeze().numpy()

dat_sub["q"] = np.minimum(p,1.-p)
dat_sub["effect_mean"] = samples["asb"].mean(0).squeeze().numpy()
dat_sub["effect_std"] = samples["asb"].std(0).squeeze().numpy()

fdr_threshold = 0.05
plt.scatter(dat_sub.input_ratio, dat_sub.IP_ratio,alpha=0.1, color="gray")
dat_ss = dat_sub[dat_sub.q < fdr_threshold]
plt.scatter(dat_ss.input_ratio, dat_ss.IP_ratio,alpha=0.03, color = "red")
plt.xlabel("Input proportion alt"); plt.ylabel("IP proportion alt")
plt.title('%i (%.1f%%) significant %.0f%% FDR' % ((dat_sub["q"] < fdr_threshold).sum(), 100. * (dat_sub["q"] < fdr_threshold).mean(), 100 * fdr_threshold))
plt.show()

dat_sub.drop(columns = ["hue", "input_ratio", "IP_ratio"]).to_csv(results_dir / "normal_struct_results.tsv.gz", index = False, sep = "\t")

########### replicates model #####################

#both = pd.concat( [dat_sub.merge( dat_sub_2.loc[:,["variantID"]], on = "variantID"), # only shared SNPs
#                dat_sub_2.merge( dat_sub.loc[:,["variantID"]], on = "variantID")], axis = 0)
both = pd.concat( (dat_sub, dat_sub_2), axis = 0)

data_both = asb_data.ReplicateASBdata.from_pandas(both)

losses, model, guide, stats, samples = gaussian_model.fit(data_both, use_structured_guide = True)

both["IP_ratio"] = both.altCount_IP / both.totalCount_IP

asb_loc = pyro.param("asb_loc").detach().numpy() # ~= samples["asb"].mean(0).squeeze().numpy()
asb_sd = torch.sqrt(pyro.param("asb_scale_param")**2 + pyro.param("asb_corr")**2).detach().numpy() # ~= samples["asb"].std(0).squeeze().numpy()

both["effect_mean"] = asb_loc[data_both.snp_indices]
both["effect_std"] = asb_sd[data_both.snp_indices]

q = scipy.stats.norm().cdf(-np.abs(asb_loc / asb_sd))
both["q"] = q[data_both.snp_indices]
both.sort_values("q").head(20)

both.drop(columns = ["input_ratio", "IP_ratio"]).to_csv(results_dir / "rep_struct_results.tsv.gz", index = False, sep = "\t")

snp_lookup = both.loc[:,["contig","position","position_x","variantID"]].drop_duplicates().rename(columns = {"position_x" : "position_hg19"})
res = pd.DataFrame( {"variantID" : data_both.snps, "q" : q, "asb_mean" : asb_loc, "asb_sd" : asb_sd })
res = snp_lookup.merge( res, on = "variantID" )
res.to_csv(results_dir / "rep_struct_asb.tsv.gz", index = False, sep = "\t")

to_plot = res.loc[:,["variantID","q"]].merge(dat_sub.loc[:,["variantID","input_ratio","totalCount_input","IP_ratio","totalCount_IP",]], on = "variantID")


plt.scatter(to_plot.input_ratio, to_plot.IP_ratio,alpha=0.1, color="gray")
dat_ss = to_plot[to_plot.q < fdr_threshold]
plt.scatter(dat_ss.input_ratio, dat_ss.IP_ratio,alpha=0.03, color = "red")
plt.xlabel("Input proportion alt"); plt.ylabel("IP proportion alt")
plt.title('%i (%.1f%%) significant %.0f%% FDR' % ((res["q"] < fdr_threshold).sum(), 100. * (res["q"] < fdr_threshold).mean(), 100 * fdr_threshold))
plt.show()

to_plot["totalCount_IP_bin"] = pd.cut(to_plot.totalCount_IP, [30,60,100,300,1000,np.inf], precision = -1)
to_plot["totalCount_input_bin"] = pd.cut(to_plot.totalCount_input, [9,30,60,100,300,1000,np.inf], precision = -1)
to_plot["asb"] = to_plot.q < fdr_threshold
to_plot["one"] = 1.
agg = to_plot.groupby(["totalCount_IP_bin","totalCount_input_bin"]).agg({"asb" : "mean", "one" : "sum"}).reset_index()
df_wide = agg.pivot_table( index='totalCount_IP_bin', columns='totalCount_input_bin', values='asb')
plt.figure(figsize=(10,6))
sns.heatmap(df_wide, annot = True)

df_wide = agg.pivot_table( index='totalCount_IP_bin', columns='totalCount_input_bin', values='one')
plt.figure(figsize=(10,6))
sns.heatmap(df_wide, annot = True)

z = asb_loc / asb_sd
nu = np.random.normal(size=len(z))

plt.rcParams.update({'font.size': 14})
f, ax = plt.subplots(figsize=(6, 4))
plt.hist(z,log=True, bins=100, alpha=0.5, label = "Data")
plt.hist(nu,log=True, bins=100, alpha = 0.5, label = "Null")
plt.legend()
plt.xlabel("Z-score")
plt.ylabel("Count")
plt.savefig("asb.pdf")
plt.show()

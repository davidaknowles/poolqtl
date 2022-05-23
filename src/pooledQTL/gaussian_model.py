import pyro
import torch
from .pyro_utils import BetaReparam, BetaBinomialReparam
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from torch.distributions import constraints
from . import pyro_utils, asb_data

def normal_model_base(data,
          ase_scale = 1.,
          ase_t_df = 3., 
          input_count_conc = 10.,
          asb_scale = 1.,
          asb_t_df = 3., 
          IP_count_conc = 10. ): # nu ~ gamma(2,0.1)
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    ase_scale = convertr(ase_scale, "ase_scale")
    input_count_conc = convertr(input_count_conc, "input_count_conc")

    asb_scale = convertr(asb_scale, "asb_scale")
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc")
    
    ase_t_df = convertr(ase_t_df, "ase_t_df")
    asb_t_df = convertr(asb_t_df, "asb_t_df")

    ase = pyro.sample("ase", # allele specific expression
        dist.StudentT(ase_t_df, 0., ase_scale).expand([data.num_snps]).to_event(1)
    )
    
    asb = pyro.sample("asb", # allele specific binding
        dist.StudentT(asb_t_df, 0., asb_scale).expand([data.num_snps]).to_event(1)
    )

    with pyro.plate("data", data.num_snps):
        input_ratio = torch.logit(data.pred_ratio) + ase
        input_alt = pyro.sample( "input_alt", 
                            BetaBinomialReparam(torch.sigmoid(input_ratio), 
                                                input_count_conc,
                                                total_count = data.input_total_count, 
                                                eps = 1.0e-8), 
                            obs = data.input_alt_count)
        IP_alt = pyro.sample( "IP_alt", 
                    BetaBinomialReparam(torch.sigmoid(input_ratio + asb), 
                                        IP_count_conc,
                                        total_count = data.IP_total_count, 
                                        eps = 1.0e-8), 
                    obs = data.IP_alt_count)
        

def rep_model_base(data,
          ase_scale = 1.,
          ase_t_df = 3., 
          input_count_conc = 10.,
          asb_scale = 1.,
          asb_t_df = 3., 
          IP_count_conc = 10. ): # nu ~ gamma(2,0.1)
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    ase_scale = convertr(ase_scale, "ase_scale")
    input_count_conc = convertr(input_count_conc, "input_count_conc")

    asb_scale = convertr(asb_scale, "asb_scale")
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc")
    
    ase_t_df = convertr(ase_t_df, "ase_t_df")
    asb_t_df = convertr(asb_t_df, "asb_t_df")

    ase = pyro.sample("ase", # allele specific expression
        dist.StudentT(ase_t_df, 0., ase_scale).expand([data.num_snps]).to_event(1)
    )
    
    asb = pyro.sample("asb", # allele specific binding
        dist.StudentT(asb_t_df, 0., asb_scale).expand([data.num_snps]).to_event(1)
    )

    with pyro.plate("data", data.num_measurements):
        input_ratio = torch.logit(data.pred_ratio) + ase[data.snp_indices]
        input_alt = pyro.sample( "input_alt", 
                            BetaBinomialReparam(torch.sigmoid(input_ratio), 
                                                input_count_conc,
                                                total_count = data.input_total_count, 
                                                eps = 1.0e-8), 
                            obs = data.input_alt_count)
        IP_alt = pyro.sample( "IP_alt", 
                    BetaBinomialReparam(torch.sigmoid(input_ratio + asb[data.snp_indices]), 
                                        IP_count_conc,
                                        total_count = data.IP_total_count, 
                                        eps = 1.0e-8), 
                    obs = data.IP_alt_count)
        
def log_sigmoid_deriv(x): 
    return F.logsigmoid( x ) + F.logsigmoid( -x )

def normal_guide(data):
    
    def conc_helper(name, init = 5.):    
        param = pyro.param(name + "_", lambda: torch.tensor(init), constraint=constraints.positive)
        return pyro.sample(name, dist.Delta(param))
    ase_scale = conc_helper("ase_scale", init = 1.)
    input_count_conc = conc_helper("input_count_conc")
    asb_scale = conc_helper("asb_scale", init = 1.)
    IP_count_conc = conc_helper("IP_count_conc")
    
    ase_t_df = conc_helper("ase_t_df", init = 3.)
    asb_t_df = conc_helper("asb_t_df", init = 3.)
    
    z1 = pyro.sample("z1", 
                    dist.Normal(torch.zeros(data.num_snps), 
                                torch.ones(data.num_snps)).to_event(1),
                    infer={'is_auxiliary': True})
    z2 = pyro.sample("z2", 
                    dist.Normal(torch.zeros(data.num_snps), 
                                torch.ones(data.num_snps)).to_event(1),
                    infer={'is_auxiliary': True})
    ase_loc = pyro.param('ase_loc', lambda: torch.zeros(data.num_snps))
    ase_scale_param = pyro.param('ase_scale_param', 
                                   lambda: torch.ones(data.num_snps), 
                                   constraint=constraints.positive)
    ase = pyro.sample("ase",  dist.Delta(ase_loc + ase_scale_param * z1,
                                              log_density = -ase_scale_param.log()).to_event(1))
    
    asb_loc = pyro.param('asb_loc', lambda: torch.zeros(data.num_snps))
    asb_scale_param = pyro.param('asb_scale_param', 
                                   lambda: torch.ones(data.num_snps), 
                                   constraint=constraints.positive)
    asb_corr = pyro.param('asb_corr', 
                                   lambda: torch.zeros(data.num_snps))
    asb = pyro.sample('asb', dist.Delta(asb_loc + asb_corr * z1 + asb_scale_param * z2,
                                              log_density = -asb_scale_param.log()).to_event(1))
    
    return {"ase_scale": ase_scale, 
            "input_count_conc": input_count_conc, 
            "asb_scale": asb_scale, 
            "IP_count_conc": IP_count_conc, 
            "ase_t_df" : ase_t_df, 
            "asb_t_df" : asb_t_df,
            "ase": ase,
            "asb": asb
           }

def fit(data, 
        iterations = 1000,
        num_samples = 300,
        use_structured_guide = True,
        learn_concs = True,
        learn_t_dof = True): 
    
    model_base = rep_model_base if ("Replicate" in str(type(data))) else normal_model_base # better syntax for determining class?
    
    two = torch.tensor(2., device = data.device)

    model = lambda data:  model_base(data, 
         ase_scale = dist.HalfCauchy(two) if learn_concs else 1., 
         input_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.,
         asb_scale = dist.HalfCauchy(two) if learn_concs else 1., 
         IP_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.,
         ase_t_df = dist.Gamma(two,two/10.) if learn_t_dof else 3., 
         asb_t_df = dist.Gamma(two,two/10.) if learn_t_dof else 3.)

    to_optimize = ["ase_scale",
                   "input_count_conc",
                   "asb_scale",
                   "IP_count_conc",
                  "ase_t_df",
                  "asb_t_df"]

    if use_structured_guide:
        guide = normal_guide
    else: 
        guide = AutoGuideList(model)
        guide.add(AutoDiagonalNormal(poutine.block(model, hide = to_optimize)))
        guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))

    losses = pyro_utils.fit(model,guide,data,iterations=iterations)

    stats,samples = pyro_utils.get_posterior_stats(model, guide, data, num_samples = num_samples, dont_return_sites = ['input_alt','IP_alt'])
    
    print({ k:stats[k]['mean'].item() for k in to_optimize }) 
    
    return losses, model, guide, stats, samples
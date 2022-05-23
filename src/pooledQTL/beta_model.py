import pyro
import torch
from . import pyro_utils
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDelta
from pyro import poutine
from torch.distributions import constraints
import torch.nn.functional as F

def model_base(data,
          input_conc = 5.,
          input_count_conc = 10. ):
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    input_conc = convertr(input_conc, "input_conc")
    input_count_conc = convertr(input_count_conc, "input_count_conc")

    input_ratio = pyro.sample("input_ratio",
        pyro_utils.BetaReparam(data.pred_ratio, input_conc).to_event(1)
    )

    with pyro.plate("data", len(data.alt_count)):
        input_alt = pyro.sample( "input_alt", 
                            pyro_utils.BetaBinomialReparam(input_ratio, 
                                                input_count_conc,
                                                total_count = data.total_count), 
                            obs = data.alt_count)

def full_model_base(data,
          input_conc = 5.,
          input_count_conc = 10.,
          IP_conc = 5., 
          IP_count_conc = 10. ):
    
    def convertr(hyperparam, name): 
        return pyro.sample(name, hyperparam) if (type(hyperparam) != float) else torch.tensor(hyperparam, device = data.device) 

    input_conc = convertr(input_conc, "input_conc")
    input_count_conc = convertr(input_count_conc, "input_count_conc")

    IP_conc = convertr(IP_conc, "IP_conc")
    IP_count_conc = convertr(IP_count_conc, "IP_count_conc")

    input_ratio = pyro.sample("input_ratio",
        pyro_utils.BetaReparam(data.pred_ratio, input_conc).to_event(1)
    )
    
    IP_ratio = pyro.sample("IP_ratio",
        pyro_utils.BetaReparam(input_ratio, IP_conc).to_event(1)
    )

    with pyro.plate("data", data.num_snps):
        input_alt = pyro.sample( "input_alt", 
                            pyro_utils.BetaBinomialReparam(input_ratio, 
                                                input_count_conc,
                                                total_count = data.input_total_count), 
                            obs = data.input_alt_count)
        IP_alt = pyro.sample( "IP_alt", 
                    pyro_utils.BetaBinomialReparam(IP_ratio, 
                                        IP_count_conc,
                                        total_count = data.IP_total_count), 
                    obs = data.IP_alt_count)


def log_sigmoid_deriv(x): 
    return F.logsigmoid( x ) + F.logsigmoid( -x )

def guide_mean_field(data):
    
    def conc_helper(name):    
        param = pyro.param(name + "_", lambda: torch.tensor(5.), constraint=constraints.positive)
        return pyro.sample(name, dist.Delta(param))
    input_conc = conc_helper("input_conc")
    input_count_conc = conc_helper("input_count_conc")
    IP_conc = conc_helper("IP_conc")
    IP_count_conc = conc_helper("IP_count_conc")
    
    input_ratio_loc = pyro.param('input_ratio_loc', lambda: torch.zeros(data.num_snps))
    input_ratio_scale = pyro.param('input_ratio_scale', 
                                   lambda: torch.ones(data.num_snps), 
                                   constraint=constraints.positive)
    input_ratio_logit = pyro.sample("input_ratio_logit", 
                                    dist.Normal(input_ratio_loc, input_ratio_scale).to_event(1),
                                    infer={'is_auxiliary': True})
    
    input_ratio = pyro.sample( "input_ratio", 
                              dist.Delta(
                                  F.sigmoid( input_ratio_logit ), 
                                  log_density = -log_sigmoid_deriv(input_ratio_logit)).to_event(1) )
    
    IP_ratio_loc = pyro.param('IP_ratio_loc', lambda: torch.zeros(data.num_snps))
    IP_ratio_scale = pyro.param('IP_ratio_scale', 
                                   lambda: torch.ones(data.num_snps), 
                                   constraint=constraints.positive)
    IP_ratio_logit = pyro.sample("IP_ratio_logit", 
                                 dist.Normal(IP_ratio_loc, IP_ratio_scale).to_event(1),
                                infer={'is_auxiliary': True})
    IP_ratio = pyro.sample( "IP_ratio", dist.Delta(F.sigmoid( IP_ratio_logit ), 
                                                  log_density = -log_sigmoid_deriv(IP_ratio_logit)).to_event(1))
    
    return {"input_conc": input_conc, 
            "input_count_conc": input_count_conc, 
            "IP_conc": IP_conc, 
            "IP_count_conc": IP_count_conc, 
            "input_ratio": input_ratio,
            "IP_ratio": IP_ratio
           }

def structured_guide(data):
    
    def conc_helper(name):    
        param = pyro.param(name + "_", lambda: torch.tensor(5.), constraint=constraints.positive)
        return pyro.sample(name, dist.Delta(param))
    input_conc = conc_helper("input_conc")
    input_count_conc = conc_helper("input_count_conc")
    IP_conc = conc_helper("IP_conc")
    IP_count_conc = conc_helper("IP_count_conc")
    
    z1 = pyro.sample("z1", 
                    dist.Normal(torch.zeros(data.num_snps), 
                                torch.ones(data.num_snps)).to_event(1),
                    infer={'is_auxiliary': True})
    z2 = pyro.sample("z2", 
                    dist.Normal(torch.zeros(data.num_snps), 
                                torch.ones(data.num_snps)).to_event(1),
                    infer={'is_auxiliary': True})
    input_ratio_loc = pyro.param('input_ratio_loc', lambda: torch.zeros(data.num_snps))
    input_ratio_scale = pyro.param('input_ratio_scale', 
                                   lambda: torch.ones(data.num_snps), 
                                   constraint=constraints.positive)
    input_ratio_logit = pyro.sample("input_ratio_logit", 
                                    dist.Delta(input_ratio_loc + input_ratio_scale * z1,
                                              log_density = -input_ratio_scale.log()).to_event(1),
                                    infer={'is_auxiliary': True})
    input_ratio = pyro.sample( "input_ratio", 
                              dist.Delta(
                                  torch.sigmoid( input_ratio_logit ), 
                                  log_density = -log_sigmoid_deriv(input_ratio_logit)).to_event(1) )
    
    IP_ratio_loc = pyro.param('IP_ratio_loc', lambda: torch.zeros(data.num_snps))
    IP_ratio_scale = pyro.param('IP_ratio_scale', 
                                   lambda: torch.ones(data.num_snps), 
                                   constraint=constraints.positive)
    IP_ratio_corr = pyro.param('IP_ratio_corr', 
                                   lambda: torch.zeros(data.num_snps))
    IP_ratio_logit = pyro.sample('IP_ratio_logit', 
                                 dist.Delta(IP_ratio_loc + IP_ratio_corr * z1 + IP_ratio_scale * z2,
                                              log_density = -IP_ratio_scale.log()).to_event(1),
                                infer={'is_auxiliary': True})
    IP_ratio = pyro.sample( "IP_ratio", dist.Delta(torch.sigmoid( IP_ratio_logit ), 
                                                  log_density = -log_sigmoid_deriv(IP_ratio_logit)).to_event(1))
    
    return {"input_conc": input_conc, 
            "input_count_conc": input_count_conc, 
            "IP_conc": IP_conc, 
            "IP_count_conc": IP_count_conc, 
            "input_ratio": input_ratio,
            "IP_ratio": IP_ratio
           }


def fit(data, 
       learn_concs = True, 
      iterations = 1000,
      num_samples = 300,
      use_structured_guide = True
      ):

    two = torch.tensor(2., device = data.device)
    
    model = lambda data:  full_model_base(data, 
         input_conc = dist.Gamma(two,two/10.) if learn_concs else 1., 
         input_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.,
         IP_conc = dist.Gamma(two,two/10.) if learn_concs else 1., 
         IP_count_conc = dist.Gamma(two,two/10.) if learn_concs else 1.)

    to_optimize = ["input_conc",
                   "input_count_conc",
                   "IP_conc",
                   "IP_count_conc"]

    if use_structured_guide: 
        guide = structured_guide
    else: 
        guide = AutoGuideList(model)
        guide.add(AutoDiagonalNormal(poutine.block(model, hide = to_optimize)))
        guide.add(AutoDelta(poutine.block(model, expose = to_optimize)))

    losses = pyro_utils.fit(model,guide,data,iterations=iterations)

    stats, samples = pyro_utils.get_posterior_stats(model, guide, data, num_samples = num_samples, dont_return_sites = ['input_alt','IP_alt'])
    
    print({ k:stats[k]['mean'].item() for k in to_optimize }) 
    
    return losses, model, guide, stats, samples
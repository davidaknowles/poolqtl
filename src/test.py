def model(data): 
    mu = pyro.sample("mu", dist.Beta(1.,1.))
    y = pyro.sample( "y", dist.Binomial(total_count = 30,  probs = mu), obs = torch.tensor(27.))

def guide(data):
    loc = pyro.param('loc', lambda: torch.tensor(0.))
    scale = pyro.param('scale', lambda: torch.tensor(1.), constraint=constraints.positive)
    logit_mu = pyro.sample("logit_mu", dist.Normal(loc, scale), infer={'is_auxiliary': True})
    mu = pyro.sample('mu', dist.Delta(torch.sigmoid( logit_mu ),
                                     log_density= -log_sigmoid_deriv(logit_mu)))
    return({"mu" : mu})

pyro.clear_param_store()
adam = pyro.optim.Adam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO() ) 
losses=[]
for j in range(300):
    loss = svi.step(None)
    losses.append(loss)
plt.plot(losses)
stats,_ = get_posterior_stats(model,
                        guide, 
                        data)
stats

nuts_kernel = NUTS(model, adapt_step_size=True)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=300) 
mcmc.run(data)
samples = mcmc.get_samples()['beta']

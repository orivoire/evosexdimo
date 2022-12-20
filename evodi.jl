module evodi
# A model to investigate the evolution of sexual dimorphism in phenotypic plasticity
# Olivier Rivoire 2022

using PyPlot, Random, Distributions, StatsBase

# Numerical estimations of analytical expressions

function value_alpha(lf, lm, s2Df, s2Dm, s2Mf, s2Mm, T=1000)
    nf, nm = 1/(1+s2Df), 1/(1+s2Dm)
    x, af, am = 0., 0., 0.
    for t = 1:T
        af = 1/(1+lf^2*nf*(x+s2Mf))
        am = 1/(1+lm^2*nm*(x+s2Mm))
        x = .25*(af*(x+s2Mf) + am*(x+s2Mm))
    end
    return x, af, am
end 

function value_logchi(kf, km, c, s2M; s2Df=0, s2Dm=0, sex="female")
    lf, lm = 1-kf, 1-km 
    cf, cm = c, c
    s2Mf, s2Mm = s2M, s2M
    nf, nm = 1/(1+s2Df), 1/(1+s2Dm)
    x, af, am = value_alpha(lf, lm, s2Df, s2Dm, s2Mf, s2Mm)
    α = .5*(af+am)
    γ = -.5*(af*cf+am*cm)
    if sex == "female"
        logchi = .5*log(af*nf) - .5*af*nf*lf^2*(γ/(1-α)-cf)^2 
    else
        logchi = .5*log(am*nm) - .5*am*nm*lm^2*(γ/(1-α)-cm)^2 
    end
    return logchi
end

function value_Lambda(kf, km, c, s2M; qf=4, qm=Inf)   
    logchif = value_logchi(kf, km, c, s2M, sex="female")
    logchim = value_logchi(kf, km, c, s2M, sex="male")
    return min(logchif + log(qf), logchim + log(qm)) - log(2)
end

function opt_kfkm(qf, qm, c, s2M; kstar=.5, dk=0.01)
    kf_opt, km_opt, L_opt = 0., 0., value_Lambda(0., 0., c, s2M, qf=qf, qm=qm)
    for kf = 0:dk:kstar
        for km = 0:dk:kstar
            L = value_Lambda(kf, km, c, s2M, qf=qf, qm=qm)
            if L > L_opt
                kf_opt, km_opt, L_opt = kf, km, L
            end
        end
    end
    return kf_opt, km_opt, L_opt
end

function value_c_star(kstar, s2M; kmin=0, delta_c=0.001, cmax=10)
    # smallest value of c at which the two-fold cost is exceeded:
    DL_threshold = log(2)
    c = 0
    DL = value_Lambda(kstar, kmin, c, s2M) - value_Lambda(kstar, kstar, c, s2M) 
    while((DL < DL_threshold) && (c<cmax))
        c += delta_c
        DL = value_Lambda(kstar, kmin, c, s2M) - value_Lambda(kstar, kstar, c, s2M)
    end
    return c
end      

# Numerical simulations of population dynamics

function pop_dyn(pop::Array{Float64,2}, σ2D::Float64, σ2M::Float64, σ2V::Float64, c::Float64, T::Int64, xt::Float64;
                 max_size=Inf, max_kappa=0.5, qfem=4, qmale=Inf, σ2e=0., chi_male=false)
    # Population dynamics 
    # initial conditions: 
    N = size(pop)[1]                                                    
    Λ, alive = 0, true
    # iterations:
    for t = 1:T
        xt += c
        pop = split_groups(pop, max_size)
        alive, pop, Wt = next_gen(pop, σ2D, σ2M, σ2V, xt, max_kappa=max_kappa, qfem=qfem, qmale=qmale, σ2e=σ2e, chi_male=chi_male)  
        if !alive # stops if extinction
            print("extinction at t=$t\n")
            break
        end                                                   
        Λ += log(Wt)
        pop = pop[rand(1:size(pop)[1], N),:] # resampling                                 
    end
    return alive, Λ/T, pop, xt
end  

function split_groups(pop, max_size; jgroup=5)
    # split a group in 2 if its size exceeds max_size
    # note that it does not guarantee that the final groups have all size < max_size
    groups = Set(pop[:,jgroup])
    for g in groups
        i_g = findall(x->x==g, pop[:,jgroup])
        if length(i_g) > max_size
            id_new_gr = minimum([x for x = 1:(length(groups)+1) if !(x in groups)])
            size_new_gr = floor(Int, length(i_g)/2)
            indices_new_gr = sample(i_g, size_new_gr, replace=false)
            pop[indices_new_gr,jgroup] = id_new_gr*ones(size_new_gr)        
            push!(groups, id_new_gr)
        end    
    end
    return pop
end

function next_gen(pop::Array{Float64,2}, σ2D::Float64, σ2M::Float64, σ2V::Float64, xt::Float64; 
                         max_kappa=.5, qfem=4, qmale=Inf, σ2e=0., chi_male=false)
    # Update the population by one generation
    # pop[i,1] = +1 if female, -1 if male (sex)
    # pop[i,2] = γ_i (main trait)
    # pop[i,3] = x_i such that κf_i = 1/(1+exp(x_i)) 
    # pop[i,4] = y_i such that κm_i = 1/(1+exp(y_i)) 
    # pop[i,5] = identity of the group (must be integers)   
    σMut = [0., sqrt(σ2M), sqrt(σ2V), sqrt(σ2V), 0.]
    groups = [x for x in Set{Int}(pop[:,5])]
    # population size and number of genes:
    N = size(pop)[1] 
    # development:
    pop_ϕ = zeros(N)
    for i = 1:N
        kappa = max_kappa / (1+exp(pop[i,3+(pop[i,1]==-1)]))
        σDeff = sqrt(σ2D + (1-kappa)^2 * σ2e)
        pop_ϕ[i] =  (1-kappa) * pop[i,2] + kappa * xt + rand(Normal(0, σDeff))
     end
    # selection:
    i_sel = [i for i = 1:N if rand() < exp(-.5*(pop_ϕ[i]-xt)^2)]                
    i_fem  = i_sel[findall(x->x==1,  pop[i_sel,1])]
    i_male = i_sel[findall(x->x==-1, pop[i_sel,1])]                                                            
    # parents with mating by group (some or all groups may become extinct):
    par_f, par_m, new_groups = [], [], []
    for g in groups
        i_f_g  = findall(x->x==g, pop[i_fem, 5])
        i_m_g  = findall(x->x==g, pop[i_male,5])
        N_f_g, N_m_g = length(i_f_g), length(i_m_g)
        if N_f_g > 0 && N_m_g > 0
            N_couple = Int(min(qfem*N_f_g, qmale*N_m_g))
            if qfem != Inf
                par_f = vcat(par_f, sample(repeat(i_fem[i_f_g], qfem), N_couple, replace=false))
            else                
                par_f = vcat(par_f, sample(i_fem[i_f_g], N_couple, replace=true))
            end
            if qmale != Inf
                par_m = vcat(par_m, sample(repeat(i_male[i_m_g], qmale), N_couple, replace=false))
            else                
                par_m = vcat(par_m, sample(i_male[i_m_g], N_couple, replace=true))
            end    
            push!(new_groups, g)
        end
    end
    # keeping only the groups with both males and females
    groups = new_groups
    N_o = length(par_f)
    if N_o == 0
        return false, zeros(0,5), 0
    else # each female makes k kids with independently chosen males
        pop_o = zeros(N_o, 5) 
        pop_o[:,1] = rand([1,-1], N_o) 
        for g in [2,3,4]             
            pop_o[:,g] = .5*(pop[par_f,g] + pop[par_m,g]) + rand(Normal(0, σMut[g]), N_o) 
        end                                  
        pop_o[:,5] = pop[par_f,5]            
    end
    if chi_male
        return true, pop_o, length(i_male)/sum(pop[:,1].==-1)
    else
        return true, pop_o, N_o/N
    end
end 

function k2y(k; max_kappa=.5, xmax=1e9)
    # inverting the relation kappa=kappa_mx/(1+exp(y))                                    
    x = 0.
    k /= max_kappa
    if k == 0
        x = xmax
    elseif k == 1
        x = -xmax
    else
        x =log(1/k-1)
    end
    return x
end                                    
                                    
function init_pop(N, kf, km; max_kappa=.5, g0=0.)
    # initialization of a population
    # all same genotype, given plasticity, single group, random sex
    pop = zeros(N,5)
    pop[:,1] = rand([-1,1],N)
    pop[:,2] = g0*ones(N)
    pop[:,3] = k2y(kf, max_kappa=max_kappa) * ones(N)
    pop[:,4] = k2y(km, max_kappa=max_kappa) * ones(N)
    return pop
end                                    
                                    
function run_dyn(c, kappa, σ2M, σ2V, N, T, tau; g0=0., σ2D=0.,
                 max_size=Inf, max_kappa=.5, qfem=4, qmale=Inf, σ2e=0.)
    # trajectories for the mean values of the reaction norms                                    
    pop = init_pop(N, kappa, kappa, max_kappa=max_kappa, g0=0.)
    kf, km = mean(max_kappa ./(1 .+exp.(pop[:,3]))), mean(max_kappa ./ (1 .+exp.(pop[:,4])))                                   
    stat = [[kf, km]]                        
    alive, xt = true, 0.
    for i = 1:floor(Int,T/tau)
        if alive
            alive, Λ, pop, xt = pop_dyn(copy(pop), σ2D, σ2M, σ2V, c, tau, xt, 
                    max_size=max_size, max_kappa=max_kappa, qfem=qfem, qmale=qmale, σ2e=σ2e)
            kf, km = mean(max_kappa ./(1 .+exp.(pop[:,3]))), mean(max_kappa ./ (1 .+exp.(pop[:,4])))
            push!(stat, [kf, km])
        end
    end                                   
    if alive                                    
        return hcat(stat...)
    else
        return NaN
    end                                    
end

function simu_Lambda(kf, km, σ2M, c, N, T, Nstat; σ2D = 0., σ2V = 0., xt=0.,
                    max_kappa=.5, qfem=4, qmale=Inf, chi_male=false)
    pop = init_pop(N, kf, km)
    Λ = [pop_dyn(copy(pop), σ2D, σ2M, σ2V, c, T, 0., max_kappa=max_kappa, qfem=qfem, qmale=qmale, chi_male=chi_male)[2] 
         for s = 1:Nstat]
    return mean(Λ), std(Λ)/sqrt(Nstat)
end                                           

function simu_Lambda_write(c, σ2M, kf_values, km_values, T, N, Nstat, filename; 
                             σ2D=0., σ2V=0., qmale=Inf, lbda1=false, qfem=4, max_kappa=.5, chi_male=false)
    open("_Outputs/"*filename*".txt", "w") do f
        write(f,"c s2M N T mean(L) std(L)\n")
        for kf in kf_values
            for km in km_values
                L, L_err = simu_Lambda(kf, km, σ2M, c, N, T, Nstat)
                write(f,"$c $(σ2M) $N $T $(kf) $(km) $(round(L,digits=4)) $(round(L_err,digits=4))\n")
            end
        end
    end
end

function simu_Lambda_read(filename)
    filename = "_Outputs/"*filename*".txt"
    L_dict, L_err_dict = Dict(), Dict()
    open(filename) do f
        for (i,line) in enumerate(eachline(f))
            if i > 1
                kf = parse(Float64, split(line)[5])
                km = parse(Float64, split(line)[6])
                L_dict[kf,km]  = parse(Float64, split(line)[7])
                L_err_dict[kf,km]  = parse(Float64, split(line)[8])
            end
        end
    end
    return L_dict, L_err_dict
end  
                                    
                                    
function simu_kappa_write(c, s2M, s2V, K_values, N, T, tau, kappa, Nstat, filename; qmale=Inf, σ2e=0.)
    open("_Outputs/"*filename*".txt", "w") do f
        write(f,"c s2M s2V K N T kf km\n")
        for K in K_values
            for s in 1:Nstat
                res = run_dyn(c, kappa, s2M, s2V, N, T, tau, max_size=K, qmale=qmale, σ2e=σ2e)
                for n = 1:floor(Int, T/tau)
                    kf, km = round.(res[1:2,n+1], digits=4)
                    write(f,"$c $(s2M) $(s2V) $K $N $(n*tau) $(kf) $(km)\n")
                end
            end
        end
    end
end

function simu_kappa_read(filename, K_values, T, Nstat)
    res_km, res_kf = Dict(K=>[] for K in K_values), Dict(K=>[] for K in K_values)
    filename = "_Outputs/"*filename*".txt"
    open(filename) do f
        for (i,line) in enumerate(eachline(f))
            if i > 1
                time = parse(Int, split(line)[6])
                if time == T
                    K = parse(Float64, split(line)[4])
                    kf = parse(Float64, split(line)[7])
                    km = parse(Float64, split(line)[8])
                    push!(res_kf[K], kf)
                    push!(res_km[K], km)
                end
            end
        end
    end
    return res_kf, res_km
end                                    

                             
                                    
end
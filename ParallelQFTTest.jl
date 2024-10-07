using ITensors, ITensorMPS, LinearAlgebra, Plots, FFTW, HDF5
n = 12
N = 2^n
r = 3
R = 2^r

#If importing, comment out sites= under #QFT
print("Loading... ")
f = h5open("/home/hayde/coding/projects/QFBE/QFT_MPO_Size"*string(n)*"by"*string(r)*"_Cutoff1E10.h5","r")
H = read(f,"H",MPO)
close(f)
sites_imported = siteinds(H)
sites = Index{Int64}[] #only need the unprimed sites
print(sites_imported)
for x=1:n+r
    push!(sites, sites_imported[x][2])
end
print("Done!")

#Create Signal
Ts = 1 / (N-1)
t0 = 0 
tmax = t0 + (N-1) * Ts
t = t0:Ts:tmax
# signal = repeat(sin.(401*2pi.*t).*sin.(190*2pi.*t),1,R)
signal = zeros(length(t),R)
signal[:,1] = sin.(100*2pi.*t)
signal[:,2] = sin.(40*2pi.*t) 
signal[:,3] = sin.(40*2pi.*t)
signal[:,4] = sin.(500*2pi.*t)
signal[:,5] = sin.(500*2pi.*t)
signal[:,6] = sin.(500*2pi.*t)
signal[:,7] = sin.(200*2pi.*t) 
signal[:,8] = sin.(50*2pi.*t)
p1 = plot(t, signal[:,1], title = "Signal") 

#FFT
F = fft(signal,1)
freqs = fftfreq(length(t), 1.0/Ts)
p2 = plot(freqs, abs2.(F[:,1]), title = "FFT")

#QFT
psi = MPS(signal,sites;cutoff=1E-10)
MPS_QFT = apply(H,psi;cutoff=1E-10)
sites_X = (reverse(sites[1:n]))#,sites[n+1:r])
sites_Y = sites[n+1:n+r]
Contracted_MPS_QFT = contract(MPS_QFT)
MPS_QFT_vec = reshape(vec(Array(Contracted_MPS_QFT,(sites_Y,sites_X))),R,N)
QFT_Result = abs2.(MPS_QFT_vec[1,:])
p3 = plot(freqs, QFT_Result, title = "QFT")

plot(p1, p2, p3, layout=(3,1), legend=false)

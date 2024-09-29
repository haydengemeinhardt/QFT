using ITensors, ITensorMPS, LinearAlgebra, Plots, FFTW, HDF5
n = 12
N = 2^n
r = 3
R = 2^r

#If importing, comment out sites= under #QFT
print("Loading... ")
f = h5open("/home/hayde/coding/projects/QFBE/QFT_MPO_Size"*string(n)*"_Cutoff1E10.h5","r")
H = read(f,"H",MPO)
close(f)
sites_imported = siteinds(H)
sites = Index{Int64}[] #only need the unprimed sites
print(sites_imported)
for x=1:n
    push!(sites, sites_imported[x][2])
end
print("Done!")

#Create Signal
Ts = 1 / (N-1)
t0 = 0 
tmax = t0 + (N-1) * Ts
t = t0:Ts:tmax
# signal = transpose(repeat(sin.(401*2pi.*t).*sin.(190*2pi.*t),1,R))
signal = zeros(length(t),R)
signal[:,1] = sin.(50*2pi.*t)
signal[:,2] = sin.(40*2pi.*t) 
signal[:,3] = sin.(40*2pi.*t)
signal[:,4] = sin.(400*2pi.*t)
signal[:,5] = sin.(300*2pi.*t)
signal[:,6] = sin.(400*2pi.*t)
signal[:,7] = sin.(200*2pi.*t) 
signal[:,8] = sin.(20*2pi.*t)
signal = transpose(signal)
p1 = plot(t, signal[1,:], title = "Signal") 

F = fft(signal,2)
freqs = fftfreq(length(t), 1.0/Ts)
p2 = plot(freqs, abs2.(F[1,:]), title = "FFT")

#QFT
sites_Y = Index(R)
A = ITensor(signal, sites, sites_Y)
U, S, V = svd(A,sites_Y)
V = S*V
psi = MPS(V,sites;cutoff=1E-10)
MPS_QFT = apply(H,psi;cutoff=1E-10)
Contracted_MPS_QFT = U*contract(MPS_QFT)
MPS_QFT_vec = reshape(Array(Contracted_MPS_QFT,(sites_Y, reverse(sites))),R,N)
QFT_Result = abs2.(MPS_QFT_vec[1,:])
p3 = plot(freqs, QFT_Result, title = "QFT")

# QFT_FBE = zeros(nb_frames_per_file,R)
# nfft = nextpow(2,size(signal,2))
# fs = N
# frame_idx = 1
# #Get the band in binary format split into an array, then truncate the array to only the needed bits
# #For example, if N=2^7=128, then the frequency range is 64Hz (Nyquist Frequency)
# #And if your frequency band is 32Hz to 64Hz, then you only need to measure the probability of the first 2 qubits being |11>
# #since all binary values between 64 and 32 start with |11>
# for band_idx=1:nb_bands
#     lim_inf = Int32(round(freq_inf[band_idx] / (fs/2) * nfft / 2)) #1000/500*
#     lim_sup = Int32(round(freq_sup[band_idx] / (fs/2) * nfft / 2))-1
#     # print("\n Infs for:",lim_inf," ")
#     QFT_FBE_infs = Prob_Freq_Above_Thresh(MPS_QFT, lim_inf, U)
#     # print("\n Sups for:",lim_sup," ")
#     QFT_FBE_sups = Prob_Freq_Above_Thresh(MPS_QFT, lim_sup, U)
#     QFT_FBE[frame_idx,:] = QFT_FBE_infs - QFT_FBE_sups
#     # print("\nQFT_FBE $lim_inf: ", QFT_FBE_infs[1])
#     # print("\nQFT_FBE $lim_sup: ", QFT_FBE_sups[1])
#     print("\nQFT_FBE $lim_inf to $lim_sup: ", QFT_FBE[1,:]*2/nfft)
# end


# #FFT
# F = fftshift(fft(signal,2),2)
# freqs = fftshift(fftfreq(length(t), 1.0/Ts),1)
# p2 = plot(freqs, abs2.(normalize(F))[1,:], title = "FFT")

# freq_inf = [1 2 10 50 200 500 1000]# 2500]
# freq_sup = [2048 10 50 200 500 1000 2048]# 5000]
# nb_bands = length(freq_inf)
# for band_idx = 1 : nb_bands #FBE analysis
#     #print("\nBand index: ", band_idx)
#     lim_inf = Int32(freq_inf[band_idx])
#     lim_sup = Int32(freq_sup[band_idx])
#     Fbe_Trace = sum(abs2.(normalize(F[1,:]))[lim_inf+2048:lim_sup+2048]); #averaging over a bin
#     #Fbe_Trace = reshape(Fbe_Trace, (size(Fbe_Trace)...,1))
#     print("\nFFT_FBE $lim_inf to $lim_sup: ", Fbe_Trace)
# end

# p3 = plot(freqs, QFT_Result, title = "QFT")
print("\nSuccessful run")
plot(p1, p2, p3, layout=(3,1), legend=false)
## import packages
using MAT
using Plots
# using MATLAB
using Dates
const USE_GPU=false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using TimerOutputs
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end
include("./seismic2D_function.jl");
## timing
ti=TimerOutput();
## define model parameters
nx=200;
nz=200;
dt=10^-3;
dx=10;
dz=10;
nt=1000;

X=(1:nx)*dx;
Z=-(1:nz)*dz;

vp=@ones(nx,nz)*2000;

# PML layers
lp=20;

# PML coefficient, usually 2
nPML=2;

# Theoretical coefficient, more PML layers, less R
# Empirical values
# lp=[10,20,30,40]
# R=[.1,.01,.001,.0001]
Rc=.0001;

# generate empty density
rho=@ones(nx,nz)*1;

# Lame constants for solid
mu=rho.*(vp/sqrt(3)).^2;
lambda=rho.*vp.^2;
## assign stiffness matrix and rho
mutable struct C2
    C11
    C13
    C33
    C55
    rho
end
C=C2(lambda+2*mu,lambda,lambda+2*mu,mu,rho);
## source
# source location
# multiple source at one time or one source at each time
msot=0;

# source location grid
s_s1=zeros(Int32,1,2);
s_s1[1]=50;
s_s1[2]=100;
s_s3=zeros(Int32,1,2);
s_s3[1]=Int(round(nz/2));
s_s3[2]=Int(round(nz/2));

# source locations true
s_s1t=dx .*s_s1;
s_s3t=maximum(Z) .-dz .*s_s3;

# magnitude
M=2.7;
# source frequency [Hz]
freq=5;

# source signal
singles=rickerWave(freq,dt,nt,M);

# give source signal to x direction
s_src1=zeros(Float32,nt,1);
s_src1=1*repeat(singles,1,length(s_s3));

# give source signal to z direction
s_src3=copy(s_src1);
s_src3=1*repeat(singles,1,length(s_s3));

# source type. 'D' for directional source. 'P' for P-source.
s_source_type=["D" "P"];
## receiver
# receiver locations grid
r1=ones(Int32,1,10);
r1[:]=30:10:120;

r3=ones(Int32,1,10);
r3[:] .= 30;

# receiver locations true
r1t=dx .*r1;
r3t=maximum(Z) .-dz .*r3;
## plot
# point interval in time steps, 0 = no plot
plot_interval=200;
# save wavefield
save_wavefield=0;
## create folder for saving
p2= @__FILE__;
if isdir(chop(p2,head=0,tail=3))==0
    mkdir(chop(p2,head=0,tail=3));
end;
## initialize seismograms
R1=zeros(Float32,nt,length(r3));
R3=copy(R1);
##
@time begin
    if msot==1
        # source locations
        s1=s_s1;
        s3=s_s3;

        # path for this source
        path=string(chop(p2,head=0,tail=3),"/multiple_source/");
        if isdir(string(path))==0
            mkdir(string(path));
        end;
        if isdir(string(path,"/rec/"))==0
            mkdir(string(path,"/rec/"));
        end;

        s1=s_s1;
        s3=s_s3;

        s1t=s_s1t;
        s3t=s_s3t;

        src1=s_src1;
        src3=s_src3;
        source_type=s_source_type;

        # pass parameters to solver
        v1,v3,R1,R3,P=VTI_2D(dt,dx,dz,nt,nx,nz,
        X,Z,
        r1,r3,
        s1,s3,src1,src3,source_type,
        r1t,r3t,
        s1t,s3t,
        lp,nPML,Rc,
        C,
        plot_interval,
        path,
        save_wavefield);

        data=R1;
        write2mat(string(path,"/rec/rec_1.mat"),data);
        data=R3;
        write2mat(string(path,"/rec/rec_3.mat"),data);
        data=P;
        write2mat(string(path,"/rec/rec_p.mat"),data);
    else
        for source_code=1:length(s_s3)
            # source locations
            s1=s_s1[source_code];
            s3=s_s3[source_code];

            # path for this source
            path=string(chop(p2,head=0,tail=3),"/source_code_",
            (source_code),"/");
            if isdir(string(path))==0
                mkdir(string(path));
            end;
            if isdir(string(path,"/rec/"))==0
                mkdir(string(path,"/rec/"));
            end;


            s1=s_s1[source_code];
            s3=s_s3[:,source_code];

            s1t=s_s1t[source_code];
            s3t=s_s3t[source_code];

            src1=s_src1[:,source_code];
            src3=s_src3[:,source_code];
            source_type=string(s_source_type[source_code]);

            # pass parameters to solver
            v1,v3,R1,R3,P=VTI_2D(dt,dx,dz,nt,nx,nz,
            X,Z,
            r1,r3,
            s1,s3,src1,src3,source_type,
            r1t,r3t,
            s1t,s3t,
            lp,nPML,Rc,
            C,
            plot_interval,
            path,
            save_wavefield);

            data=R1;
            write2mat(string(path,"/rec/rec_1.mat"),data);
            data=R3;
            write2mat(string(path,"/rec/rec_3.mat"),data);
            data=P;
            write2mat(string(path,"/rec/rec_p.mat"),data);
        end
    end
end

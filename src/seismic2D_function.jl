function meshgrid(x,y)
    x2=zeros(length(x),length(y));
    y2=x2;
    x2=repeat(x,1,length(y));
    y2=repeat(reshape(y,1,length(y)),length(x),1);
    return x2,y2
end

function write2mat(path,var)
    file=matopen(path,"w");
    write(file,"data",data);
    close(file);
    return 0
end

function rickerWave(freq,dt,ns,M)
    ## calculate scale
    E=10 .^(5.24+1.44 .*M);
    s=sqrt(E.*freq/.299);

    t=dt:dt:dt*ns;
    t0=1 ./freq;
    t=t .-t0;
    ricker=s .*(1 .-2*pi^2*freq .^2*t .^2).*exp.(-pi^2*freq^2 .*t .^2);
    ricker=ricker;
    ricker=Float32.(ricker);
    return ricker
end
##
@parallel function compute_sigma(dt,dx,dz,C11,C13,C33,C55,beta,v1,v1_3_2_end,v3,v3_1_2_end,
    sigmas11,sigmas13,sigmas33,p)

    @inn(sigmas11)=dt*.5*((@all(C11)-@all(C13)) .*@d_xi(v1)/dx+
    (@all(C13)-@all(C33)) .*@d_yi(v3)/dz)+
    @inn(sigmas11)-
    dt*@all(beta) .*@inn(sigmas11);

    @inn(sigmas33)=dt*.5*((@all(C33)-@all(C13)) .*@d_yi(v3)/dz+
    (@all(C13)-@all(C11)) .*@d_xi(v1)/dx)+
    @inn(sigmas33)-
    dt*@all(beta).*@inn(sigmas33);

    @inn(sigmas13)=dt*(@all(C55) .*(@d_yi(v1_3_2_end)/dz+@d_xi(v3_1_2_end)/dx))+
    @inn(sigmas13)-
    dt*@all(beta).*@inn(sigmas13);

    # p
    @inn(p)=-dt*((@all(C11)+@all(C33))*.5 .*@d_xi(v1)/dx+
    (@all(C13)+@all(C33))*.5 .*@d_yi(v3)/dz)+
    @inn(p)-
    dt*@all(beta).*@inn(p);

    return nothing
end

@parallel_indices (iz) function x_2_end(in,out)
out[:,iz]=in[2:end,iz];
return nothing
end

@parallel_indices (ix) function z_2_end(in,out)
out[ix,:]=in[ix,2:end];
return nothing
end
##
@parallel function compute_v(dt,dx,dz,rho,v1,v3,beta,sigmas11_minus_p_1_2_end,
    sigmas13,sigmas33_minus_p_3_2_end)

    @inn(v1)=dt./@all(rho) .*(@d_xi(sigmas11_minus_p_1_2_end)/dx+
    @d_yi(sigmas13)/dz)+
    @inn(v1)-
    dt*@all(beta) .*@inn(v1);

    @inn(v3)=dt./@all(rho) .*(@d_xi(sigmas13)/dx+
    @d_yi(sigmas33_minus_p_3_2_end)/dz)+
    @inn(v3)-
    dt*@all(beta) .*@inn(v3);
    return nothing
end
@parallel function minus(a,b,c)
    @all(c)=@all(a)-@all(b);
    return nothing
end

@timeit ti "VTI_2D" function VTI_2D(dt,dx,dz,nt,
    nx,nz,X,Z,r1,r3,s1,s3,src1,src3,source_type,
    r1t,r3t,
    s1t,s3t,
    lp,nPML,Rc,
    C,
    plot_interval,
    path,
    save_wavefield)

    d0=Dates.now();

    #create folder for figures
    if isdir(path)==0
        mkdir(path);
    end

    n_picture=1;
    if save_wavefield==1
        if isdir(string(path,"/forward_wavefield/"))==0;
            mkdir(string(path,"/forward_wavefield/"));
        end
    end

    if plot_interval!=0
        if isdir(string(path,"/forward_pic/"))==0
            mkdir(string(path,"/forward_pic/"))
        end
    end

    # PML
    vmax=sqrt.((C.C33) ./C.rho);
    beta0=(ones(nx,nz) .*vmax .*(nPML+1) .*log(1/Rc)/2/lp/dx);
    beta1=(@zeros(nx,nz));
    beta3=beta1;
    tt=(1:lp)/lp;
    tt2=repeat(reshape(tt,lp,1),1,nz);
    plane_grad1=@zeros(nx,nz);
    plane_grad3=plane_grad1;

    plane_grad1[2:lp+1,:]=reverse(tt2,dims=1);
    plane_grad1[nx-lp:end-1,:]=tt2;
    plane_grad1[1,:]=plane_grad1[2,:];
    plane_grad1[end,:]=plane_grad1[end-1,:];

    tt2=repeat(reshape(tt,1,lp),nx,1);
    plane_grad3[:,2:lp+1]=reverse(tt2,dims=2);
    plane_grad3[:,nz-lp:end-1]=tt2;
    plane_grad3[:,1]=plane_grad3[:,2];
    plane_grad3[:,end]=plane_grad3[:,end-1];

    beta1=beta0.*plane_grad1.^nPML;
    beta3=beta0.*plane_grad3.^nPML;

    IND=unique(findall(f-> f!=0,beta1.*beta3));
    beta=beta1+beta3;
    beta[IND]=beta[IND]/2;

    beta1=beta3=plane_grad1=plane_grad3=vmax=nothing;

    # receiver configuration
    R1=@zeros(nt,length(r3));
    R3=@zeros(nt,length(r3));
    P=@zeros(nt,length(r3));

    # wave vector
    v1=@zeros(nx,nz);
    v3=@zeros(nx,nz);

    sigmas11=@zeros(nx,nz);
    sigmas13=@zeros(nx,nz);
    sigmas33=@zeros(nx,nz);
    p=@zeros(nx,nz);

    l=1;
    # save wavefield
    if save_wavefield==1
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/v1_",l,".mat"),data);
        data=zeros(nx,nz);
        write2ma(string(path,"/forward_wavefield/v3_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/sigmas11_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/sigmas33_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/sigmas13_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/p_",l,".mat"),data);
    end
    #
    v1_3_2_end=@zeros(nx,nz-1);
    v3_1_2_end=@zeros(nx-1,nz);
    sigmas11_minus_p_1_2_end=@zeros(nx-1,nz);
    sigmas33_minus_p_3_2_end=@zeros(nx,nz-1);
    sigmas11_minus_p=@zeros(nx,nz);
    sigmas33_minus_p=@zeros(nx,nz);

    for l=1:nt-1
        @timeit ti "shift coordinate" @parallel (2:nx-1) z_2_end(v1,v1_3_2_end);
        @timeit ti "shift coordinate" @parallel (2:nz-1) x_2_end(v3,v3_1_2_end);
        @timeit ti "compute_sigma" @parallel compute_sigma(dt,dx,dz,C.C11,C.C13,C.C33,C.C55,beta,v1,v1_3_2_end,
        v3,v3_1_2_end,
        sigmas11,sigmas13,sigmas33,p);

        @timeit ti "minus" @parallel minus(sigmas11,p,sigmas11_minus_p);
        @timeit ti "minus" @parallel minus(sigmas33,p,sigmas33_minus_p);

        @timeit ti "shift coordinate" @parallel (2:nz-1) x_2_end(sigmas11_minus_p,sigmas11_minus_p_1_2_end);
        @timeit ti "shift coordinate" @parallel (2:nx-1) z_2_end(sigmas33_minus_p,sigmas33_minus_p_3_2_end);

        @timeit ti "compute_v" @parallel compute_v(dt,dx,dz,rho,v1,v3,beta,sigmas11_minus_p_1_2_end,
        sigmas13,sigmas33_minus_p_3_2_end);

        @timeit ti "source" if source_type=="D"
        v1[CartesianIndex.(s1,s3)]=v1[CartesianIndex.(s1,s3)]+
        1 ./C.rho[CartesianIndex.(s1,s3)] .*reshape(src1[l,:],1,length(s3));
        v3[CartesianIndex.(s1,s3)]=v3[CartesianIndex.(s1,s3)]+
        1 ./C.rho[CartesianIndex.(s1,s3)] .*reshape(src3[l,:],1,length(s3));
        end

    @timeit ti "source" if source_type=="P"
    p[CartesianIndex.(s1,s3)]=p[CartesianIndex.(s1,s3)]+reshape(src3[l,:],1,length(s3));
    end

    # assign recordings
    @timeit ti "receiver" R1[l+1,:]=reshape(v1[CartesianIndex.(r1,r3)],length(r3),);
    @timeit ti "receiver" R3[l+1,:]=reshape(v3[CartesianIndex.(r1,r3)],length(r3),);
    @timeit ti "receiver" P[l+1,:]=reshape(p[CartesianIndex.(r1,r3)],length(r3),);
    # save wavefield
    if save_wavefield==1
        data=v1;
        write2mat(string(path,"/forward_wavefield/v1_",l+1,".mat"),data);
        data=v3;
        write2mat(string(path,"/forward_wavefield/v3_",l+1,".mat"),data);
        data=sigmas11;
        write2mat(string(path,"/forward_wavefield/sigmas11_",l+1,".mat"),data);
        data=sigmas33;
        write2mat(string(path,"/forward_wavefield/sigmas33_",l+1,".mat"),data);
        data=sigmas13;
        write2mat(string(path,"/forward_wavefield/sigmas13_",l+1,".mat"),data);
        data=p;
        write2mat(string(path,"/forward_wavefield/p_",l+1,".mat"),data);
    end

    # plot
    l2=Float32(l);
    if plot_interval!=0
        if mod(l,plot_interval)==0 || l==nt-1
            heatmap((1:nx) .*dx .+minimum(X), reverse(maximum(Z) .-(1:nz) .*dz),v3',
            xaxis="x [m]",
            yaxis="z [m]",
            title=string("t =",l*dt,"s","\n v3[m/s]"));
            plot!(reshape(r1t,length(r1t),),
            reshape(r3t,length(r3t),),
            linestyle = :dot,linealpha = 0.5,
            linewidth = 4,linecolor = :blue,
            label="receiver");
            savefig(string(path,"./forward_pic/v3_",n_picture,".png"));
            #=
            mat"
            hfig=figure('Visible','off');
            set(hfig,'position',[0,0,900,400]);
            subplot(2,3,1)
            imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$v3');
            axis on;
            hold on;
            set(gca,'ydir','normal');
            colorbar;
            xlabel({['x [m]']});
            ylabel({['z [m]']});
            title({['t=' num2str($l2*$dt) 's'],['v_3 [m/s]']});
            xlabel('x [m]');
            ylabel('z [m]');
            colorbar;
            hold on;
            ax2=plot($s1t,$s3t,'v','color',[1,0,0]);
            hold on;
            ax4=plot($r1t,$r3t,'^','color',[0,1,1]);

            subplot(2,3,2)
            imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$sigmas33');
            axis on;
            set(gca,'ydir','normal');
            xlabel('x [m]');
            ylabel('z [m]');
            title('sigmas33 [Pa]');
            colorbar;

            subplot(2,3,3)
            imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$sigmas13');
            axis on;
            set(gca,'ydir','normal');
            xlabel('x [m]');
            ylabel('z [m]');
            title('sigmas13 [Pa]');
            colorbar;

            subplot(2,3,4)
            imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$p');
            axis on;
            set(gca,'ydir','normal');
            xlabel('x [m]');
            ylabel('z [m]');
            title('p [Pa]');
            colorbar;

            subplot(2,3,5)
            axis on;
            imagesc([1,length($r1)],[1,($l2+1)]*$dt,$R3(1:$l2+1,:));
            colorbar;
            xlabel('Nr');
            ylabel('t [s]');
            title('R3 [m/s]');
            ylim([1,$nt]*$dt);

            subplot(2,3,6)
            axis on;
            imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$C.C33');
            set(gca,'ydir','normal');
            xlabel('x [m]');
            ylabel('z [m]');
            title('C33 [Pa]');
            colorbar;
            hold on;
            ax2=plot($s1t,$s3t,'v','color',[1,0,0]);
            hold on;
            ax4=plot($r1t,$r3t,'^','color',[0,1,1]);

            legend([ax2,ax4],...
            'source','receiver',...
            'Location',[0.5,0.02,0.005,0.002],'orientation','horizontal');

            print(gcf,[$path './forward_pic/' num2str($n_picture) '.png'],'-dpng','-r200');
            "
            =#
            n_picture=n_picture+1;
        end
    end
    d=Dates.now();
    println("\n time step=",l+1,"/",nt);
    println("\n    n_picture=",n_picture);
    println("\n    ",d);
end
return v1,v3,R1,R3,P
end

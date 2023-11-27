
clear all
close all

nSamples = 5000; %number of samples 
nBurnin=1000; %number of burnin


% DATA AND PARAMETERS
% Choose according to the examples 
% data_LOGCOX, data_LOGISTIC 
% or define your data
% data_example


[XX,y,alpha,d,X]=data_LOGISTIC;
%[x,sigmainv,m,mu,d,xx,yy,cell_counts_]=data_LOGCOX;


% Inizializzazioni
q = zeros(d,nSamples);
Accepted=0; Rejected=0;
AR=[];%acceptance rate
DH=[];%delta Energy
DH_mean=[]; 

% DEFINE POTENTIAL ENERGY FUNCTION and GRADIENT 

% Choose the parameter 0< epsilon < 1 for perturbed U 
% epsilon=0 for Gaussian
% epsilon=1 for U non-perturbed 
epsilon=1;

% Choose according to the examples 
% potential_LOGISTIC, gradient_potential_LOGISTIC
% or 
% potential_LOGCOX, gradient_potential_LOGCOX
% or
% define your example
% potential_example, gradient_potential_example



 % DEFINE POTENTIAL ENERGY FUNCTION IN PERTURBED FORM

% U = potential_LOGCOX(q,epsilon,x,sigmainv,m,mu);  
 U = potential_LOGISTIC(q,epsilon,XX,y);
  

    % DEFINE GRADIENT OF POTENTIAL ENERGY IN PERTURBED FORM
   % dU = gradient_potential_LOGCOX(q,epsilon,x,sigmainv,m,mu);  
    dU =gradient_potential_LOGISTIC(q,epsilon,XX,y);



% DEFINE KINETIC ENERGY FUNCTION
K = @(p) (transpose(p)*p)/2;


% DEFINE INITIAL SAMPLE
%q0=initial_LOGCOX(mu,d);
q0=initial_LOGISTIC(alpha,d);

q(:,1) = q0;
Tmax=3;

%initial value for integrator and number of sample

bmax=0.1932; red=0.98;  %For LOGISTIC
%bmax=0.2113; red=0.997; % For LOGCOX  
bmin=(3-sqrt(5))/4;
factor=bmax-bmin;
b=bmax; delta=sqrt((4*b^2-6*b+1)/(2*b-1))/b;


Tmax=3;

bvec=[]; deltavec=[];

t = 1;


while t < nSamples
    t = t + 1

%      
     deltavec=[deltavec, delta];
     bvec=[bvec,b];
%     

    
    % SAMPLE RANDOM MOMENTUM

     p0 = normrnd(0,1,[d,1]);
 
    pStar=p0;
    qStar=q(:,t-1);
   
    % Drow random Tstar
    u=rand;
    Tstar=delta+(Tmax-delta)*u;
    LL=floor(Tstar/delta);
    %% SIMULATE HAMILTONIAN DYNAMICS
     for j = 1:LL
        pStar = pStar - b * delta * dU(qStar);
        qStar = qStar + delta/2 * pStar;
        pStar = pStar - (1 - 2 * b) * delta * dU(qStar);
        qStar = qStar + delta/2 * pStar; 
        pStar = pStar - b * delta * dU(qStar);
    end

    % EVALUATE ENERGIES AT
    % START AND END OF TRAJECTORY
    U0 = U(q(:,t-1));
    UStar = U(qStar);

    K0 = K(p0);
    KStar = K(pStar);
    H0=U0+K0;
    HStar=UStar+KStar;


    % ACCEPTANCE/REJECTION CRITERION
    alpha = min(1,exp(H0 - HStar));
    %u = rand;
    if u < alpha && (~isnan(HStar))
        q(:,t) = qStar;
        if t>nBurnin
        Accepted=Accepted+1;
        DH=[DH,abs(-H0 + HStar)];
        end
        
    else
        q(:,t) = q(:,t-1);
        
        % Comment lines 135-143 and i48 if you want to refine b at each
        % rejection

%           if t>nBurnin
%           Rejected=Rejected+1;
%           control=Rejected/(t-nBurnin);
%           if control>0.6
%           factor=red*factor;
%           b= bmin+factor;
%           delta=sqrt((4*b^2-6*b+1)/b/(2*b-1));
%           end
%           else
          factor=red*factor;
          b= bmin+factor;
          delta=sqrt((4*b^2-6*b+1)/(2*b-1))/b;
          
          %end
        
    end
end
DH_mean=[DH_mean,mean(DH)];
AR=[AR,Accepted/(nSamples-nBurnin)];


%DISPLAY b and delta values 
figure(5)
plot(bvec)
figure(6)
plot(deltavec)


%DISPLAY ACCORDING TO THE EXAMPLE

   % display_LOGCOX(q,d,xx,yy,nSamples,nBurnin,cell_counts_)
    display_LOGISTIC(q,X,y)

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,sigmainv,m,mu,d,xx,yy,cell_counts]=data_LOGCOX;

    data= importdata("alianto.csv");
    data=[data(:,1),-data(:,2)];
    d=64^2;
    ds=sqrt(d);
    % do this if you want to rescale from 0 to 1 the data
    data(:,1)=rescale(data(:,1),0.1,0.9);
    data(:,2)=rescale(data(:,2),0.1,0.9);
    %save("alianto_scaled.csv",pines_scale)
    %scatter(data(:,1),data(:,2))
    %build grid and count the number in each cell
    nx=ds;
    ny=ds;
    xx     = 0:1/(ds - 1):1;
    yy      = 0:1/(ds - 1):1;
    [xc, yc] = meshgrid(xx, yy);
    xx_ = (0-1/(2*(ds-1)):1/(ds -1):(1+1/(2*(ds-1))));
    yy_ = (0-1/(2*(ds-1)):1/(ds -1):(1+1/(2*(ds-1))));
    [xg, yg] = meshgrid(xx_, yy_);
    figure();
    plot(xg,yg,'b');
    hold on
    plot(xg',yg','b');
    scatter(data(:,1),data(:,2));
    title('plot of point on enlarge grid')
    figure()
    plot(xc,yc,'b');
    hold on
    plot(xc',yc','b');
    scatter(data(:,1),data(:,2));
    title('plot of point on grid centroid')
    cell_counts_ = flip(histcounts2(data(:, 2), data(:, 1), xx_, yy_));
    cell_counts=cell_counts_';
    x=cell_counts(:);
    %csvwrite("alianto_point.csv",x);
    %calculate distance of all centroid from others
    dy=reshape(yc,[],1);
    dx=reshape(xc,[],1);
    ss=[dx,dy];
    dist1=pdist2(ss,ss);
    var=3.5881;
    mu=log(sum(sum(x)))-(var/2);
    beta=0.127;
    m=1/(d);
    %mean=mu*eye(d);
    sigma=var.*exp(-dist1./(ds*beta));
    %decomment or comment below if you want or not the cholewsky factorization of sigmainv
    R=chol(sigma); % R is upper triangular such that Sigma=R'*R
    sigmainv=R\eye(d);
    sigmainv=sigmainv*sigmainv';
    end


    function U=potential_LOGCOX(q,epsilon,x,sigmainv,m,mu)

  
    % DEFINE POTENTIAL ENERGY FUNCTION 
    U= @(q)  (1-epsilon)/2*q'*q +epsilon*(-x'*q + m*sum(exp(q)) + 0.5*((q - mu)'*sigmainv*(q - mu)));
    end

    function dU=gradient_potential_LOGCOX(q,epsilon,x,sigmainv,m,mu);
   

    % DEFINE GRADIENT OF POTENTIAL ENERGY
  
  
    dU = @(q) (1-epsilon)*q + epsilon* (-x + (m*exp(q)) + sigmainv*(q - mu)); 
    end

   function q0=initial_LOGCOX(mu,d);
   q0 = mu*ones(d,1);
   end

   function display_LOGCOX(q,d,xx,yy,nSamples,nBurnin,cell_counts_)
    y=q;d=sqrt(d);
   
    aa=reshape(y(:,nSamples-1),d,[])';
    figure()
    %aa=flipud(aa);
    image(xx,yy,aa,'CDataMapping','scaled')
    title('Hamiltonian Monte Carlo last plot of sample')
    eff_sample=y(:,nBurnin:nSamples);
    expected=mean(eff_sample,2);
    ee=reshape(expected,d,d)';

    figure()
    image(xx,yy,ee,'CDataMapping','scaled')
    title('Hamiltonian Monte Carlo mean of all sample final plot')
    y = 0:0.1:1;
    yTickLabels = arrayfun(@num2str,sort(y,'descend'),'uni',false);
    ax          = gca;
    ax.YAxis.TickLabels = yTickLabels;
    xlabel('X Position');
    ylabel('Y Position');
    colorbar

    figure()
    image(xx,yy,cell_counts_,'CDataMapping','scaled')
    colorbar
    shading('interp')
    colormap('jet')
    title('number of point heat map')

    figure()
    pcolor(xx,yy,flip(cell_counts_))
    shading('interp')
    colormap('jet')
    title('number of point heat map shaded')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    function [XX,y,alpha,d,X]=data_LOGISTIC
   
    Polynomial_Order = 1;
    alpha=100;
    %Load and prepare train & test data
    load('pima.mat');
    y=X(:,end);
    X(:,end)=[];

    % Normalise data
    [N, d] = size(X);
    

    X = (X/sqrt(alpha)-repmat(mean(X/sqrt(alpha)),N,1))./repmat(std(X/sqrt(alpha)),N,1);


    %Create Polynomial Basis
    XX = ones(size(X,1),1);
    for i = 1:Polynomial_Order
        XX = [XX X.^i];
    end
    [N,d] = size(XX);
    
    alpha=1;
    end
 


    function U=potential_LOGISTIC(q,epsilon,XX,y)

   
    % DEFINE POTENTIAL ENERGY FUNCTION IN PERTURBED FORM
    
    U = @(q) (1-epsilon)/2*q'*q +epsilon*(0.5*q'*q+ (-q'*XX'*y + sum(log((1+exp(XX*q))))));
    end

    function dU=gradient_potential_LOGISTIC(q,epsilon,XX,y)

    % DEFINE GRADIENT OF POTENTIAL ENERGY IN PERTURBED FORM
    dU = @(q) (1-epsilon)*q + epsilon* (q - XX'*( y - (exp(XX*q)./(1+exp(XX*q)))));
    end  

    function q0=initial_LOGISTIC(alpha,d)

    q0 = normrnd(0,sqrt(alpha),[d,1]);
    end

    function display_LOGISTIC(q,X,y)
     mdl = fitglm(X,y,'Distribution','binomial');
     estimateglm=mdl.Coefficients.Estimate;


    figure()
 tiledlayout(2,4)
 for k=1:8
 nexttile
 histogram(q(k,100:1000),'Normalization','probability', 'NumBins',30)
 title(['\beta_',num2str(k-1)])
 hold on
 xline(estimateglm(k), 'Color','red',LineWidth=2.3)
 end
    end
   

   
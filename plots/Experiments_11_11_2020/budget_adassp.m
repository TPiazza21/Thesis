function [thetahat] =  budget_adassp(X,y, opts)
    % this is Tyler's adaptive version, where we have gamma, toggling how
    % much privacy budget is spread
    BX = 1;
    BY = 1;
   
    epsilon=opts.eps;
    delta=opts.delta;
    gamma = opts.gamma;
    
    hyper_epsilon = epsilon * gamma;
    hyper_delta = delta * gamma;
    
    main_epsilon = (1-gamma) * epsilon / 2.0;
    main_delta = (1-gamma) * epsilon / 2.0;
    
    [n,d]= size(X);
        
    varrho=0.05;
    
    % this is the hyperparameter section, so use special epsilon and gamma
    % set the eigenvalue limit
    eta = sqrt(d*log(6/hyper_delta)*log(2*d^2/varrho))*BX^2/(hyper_epsilon/3);
    
    XTy = X'*y;
    XTX = X'*X + eye(d);
    
    S=svd(XTX);
    S=diag(S);
    logsod = log(6/hyper_delta);
    
    lamb_min = S(end)+ randn()*BX^2*sqrt(logsod)/(hyper_epsilon/3) - logsod/(hyper_epsilon/3);%eigs(XTX,1,'sa');
    lamb_min = max(lamb_min,0);
    
    lamb = max(0,eta - lamb_min);
    
    
    % this is the "main" stuff, so use main epsilon and delta
    XTyhat = XTy + (sqrt(log(6/main_delta))/(main_epsilon/3))*BX*BY*randn(d,1);
    % generate symmetric gaussian noise
    Z=randn(d,d);
    Z=0.5*(Z+Z');
    XTXhat = XTX + (sqrt(log(6/main_delta))/(main_epsilon/3))*BX*BX*Z;
    
    thetahat = (XTXhat+lamb*eye(d))\XTyhat;
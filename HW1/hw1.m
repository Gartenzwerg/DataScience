mu = [0 0];
sigma = [1 0.9999; 0.9999 1];
% xx = TwoDNormal(mu,sigma,50);
beta = [-1 2]';
% yy = Response(xx,beta,0,1,50);

%fitlm(xx,yy)
% OLS(xx,yy)
%ridge(yy,xx,0.005)
% RidgeRegression(xx,yy,0.005)

betas = zeros(2000,4);

for iter = 1:2000
    xx = TwoDNormal(mu,sigma,50);
    yy = Response(xx,beta,0,1,50);
    betas(iter,1:2) = OLS(xx,yy);
    betas(iter,3:4) = RidgeRegression(xx,yy,0.005);
end

win = 0;
for iter = 1:2000
    a1 = abs(-1-betas(iter,1));
    a2 = abs(-1-betas(iter,3));
    if a2 > a1
        win = win+1;
    end
end

win

f1 = figure;
histogram(betas(:,1),'BinWidth',2,'BinLimits',[-42,42])
% hold on;
f2 = figure;
histogram(betas(:,3),'BinWidth',2,'BinLimits',[-42,42])
f3 = figure;
histogram(betas(:,1),'BinWidth',2,'BinLimits',[-42,42])
hold on;
histogram(betas(:,3),'BinWidth',2,'BinLimits',[-42,42])

function X = TwoDNormal(MU, SIGMA, num)
    for i = 1:num
        X(i,:) = mvnrnd(MU,SIGMA);
    end
end

function Y = Response(X,beta,error_mean,error_var,num)
    Y = X*beta;
    for i = 1:num
        error = random('Normal',error_mean,error_var);
        Y(i) = Y(i)+ error;
    end
end

function beta_ols = OLS(XX,Y)
    [m,n] = size(XX);
    ones(1:m,1)= 1;
    X = [ones XX];
    beta_ols = (X'*X\X')*Y;
    beta_ols = beta_ols(2:3);
end


function beta_ridge = RidgeRegression(X, Y, lambda)
    [m,n] = size(X);
    for j = 1:n
        mn = mean(X(:,j));
        for i = 1:m
            X(i,j) = X(i,j) - mn;
        end
    end
    M = (X'*X+lambda*eye(n))\X';
    beta_ridge = M*Y;
end
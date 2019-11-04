M = csvread('HW5_dat.csv',1);
X = transpose(M(:,1:2));
n=size(X,2);
% figure;
% for i=1:n
%     if M(i,3)==1
%         scatter(M(i,1),M(i,2),36,'blue');
%     else
%         scatter(M(i,1),M(i,2),36,'red');
%     end
%     hold on;
% end
% hold off;
X_centered = X*(eye(n)-ones(n,1)*ones(1,n)/n);

% % PCA
K = X_centered.'*X_centered;
[Sigma1,V1,Y1]=KernelPCA(K,2);
% figure;
% for i=1:n
%     if M(i,3)==1
%         scatter(Y1(1,i).',Y1(2,i).',36,'blue');
%     else
%         scatter(Y1(1,i).',Y1(2,i).',36,'red');
%     end
%     hold on;
% end
% hold off;

% GaussianKernel
KK = KappaMatrix(X,'GaussianKernel',0.2);
[Sigma2,V2,Y2]=KernelPCA(KK,2);
% figure;
% for i=1:n
%     if M(i,3)==1
%         scatter(Y2(1,i).',Y2(2,i).',36,'blue');
%     else
%         scatter(Y2(1,i).',Y2(2,i).',36,'red');
%     end
%     hold on;
% end
% hold off;
% figure;
% scatter(Y(1,:).',Y(3,:).');
% figure;
% scatter(Y(2,:).',Y(3,:).');

% PolynomailKernel
KK = KappaMatrix(X,'Polynomial',2);
[Sigma3,V3,Y3]=KernelPCA(KK,3);
% figure;
% for i=1:n
%     if M(i,3)==1
%         scatter(Y3(1,i).',Y3(2,i).',36,'blue');
%     else
%         scatter(Y3(1,i).',Y3(2,i).',36,'red');
%     end
%     hold on;
% end
% hold off;
figure;
for i=1:n
    if M(i,3)==1
        scatter(Y3(1,i).',Y3(3,i).',36,'blue');
    else
        scatter(Y3(1,i).',Y3(3,i).',36,'red');
    end
    hold on;
end
hold off;
figure;
figure;
for i=1:n
    if M(i,3)==1
        scatter3(Y3(1,i).',Y3(2,i).',Y3(3,i).',36,'blue');
    else
        scatter3(Y3(1,i).',Y3(2,i).',Y3(3,i).',36,'red');
    end
    hold on;
end
hold off;

% New data points
Xnew = [0 0 0.7 -0.7; -0.7 0.7 0 0];

K1 = Kappa_new(X_centered, Xnew, 'PCA', 0);
K2 = Kappa_new(X_centered, Xnew, 'GaussianKernel',0.2);
K3 = Kappa_new(X_centered, Xnew, 'Polynomial',2);

Ynew1 = Sigma1\V1.'*K1;
Ynew2 = Sigma2\V2.'*K2;
Ynew3 = Sigma3\V3.'*K3;

figure;
scatter([1 2 3 4],Ynew1(1,:).');
figure;
scatter(Ynew2(1,:).',Ynew2(2,:).');
figure;
scatter(Ynew3(1,:).',Ynew3(2,:).');

function G = GaussianKernel(x,y,sigma)
    G = exp(-((norm(x-y)/sigma)^2)/2);
end

function P = PolynomialKernel(x,y,a)
    P = dot(x,y)^a;
end

function K = KappaMatrix(X,type,tuning_parameter)
  p = size(X,1);
  n = size(X,2);
  H = eye(n)-ones(n,1)*ones(1,n)/n;
  if type == "Polynomial"
      X = X*H;
  end
  K = zeros(n,n);
  for i=1:n
      xx = X(:,i);
      for j=i:n
         yy = X(:,j);
         if type == "GaussianKernel"
           K(i,j)=GaussianKernel(xx,yy,tuning_parameter);
         elseif type == "Polynomial"
           K(i,j)=PolynomialKernel(xx,yy,tuning_parameter);
         end
         K(j,i)=K(i,j);
      end     
  end
  K = H*K*H;
end

function [Sigma,V,Y] = KernelPCA(kappa,d)
    [U,S,V] = svd(kappa);
    V = V(:,1:d);
    Sigma = sqrtm(S(1:d,1:d));
    Y = Sigma*V.';
end

function K = Kappa_new(X_old_centered,X_new_centered, type, tuning_parameter)
    n = size(X_old_centered, 2);
    l = size(X_new_centered, 2);
    K = zeros(n,l);
    for i = 1:n
        xx = X_old_centered(:,i);
        for j = 1:l
            yy = X_new_centered(:,j);
            if type == "GaussianKernel"
                K(i,j)=GaussianKernel(xx,yy,tuning_parameter);
            elseif type == "Polynomial"
                K(i,j)=PolynomialKernel(xx,yy,tuning_parameter);
            elseif type == "PCA"
                K(i,j)=dot(xx,yy);
            end
        end
    end
end
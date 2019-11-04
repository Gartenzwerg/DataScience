D = xlsread('US_distance.xlsx',1);

[Y,E] = ClassicalMDS(D,2);

figure;
s = sum(abs(E));
E = E/s;
plot(E);

figure;
scatter(Y(1,:),Y(2,:));

D2 = xlsread('US_distance.xlsx',2);
[Y2,E2] = ClassicalMDS(D2,2);
figure;
scatter(-Y2(1,:),-Y2(2,:));

D3 = xlsread('US_distance.xlsx',3);
[Y3,E3] = ClassicalMDS(D3,2);
figure;
scatter(Y3(1,:),-Y3(2,:));

function [Y,E] = ClassicalMDS(D,r)
    D2 = D.^2;
    n = size(D2,1);
    H = eye(n)-ones(n)*ones(n)'/n;
    D2 = -H*D2*H./2;
    [V,S] = eig(D2);
    [E,I] = maxk(diag(S),r);
    VV = zeros(n,r);
    EE = zeros(r,r);
    for i = 1:r
        VV(:,i) = V(:,I(i));
        EE(i,i) = sqrt(E(i));
    end
    Y = EE*VV.';
    E = sort(diag(S),'descend');
end
M = csvread('HW9.csv',2);
n = size(M,1);
p = size(M,2)-1;
X = M(1:n,1:p);
Y = M(1:n,p+1);

% Hard margin SVM
HardMarginSVM = fitcsvm(X,Y,'BoxConstraint',0.02);
sv = HardMarginSVM.SupportVectors;
figure;
gscatter(X(:,1),X(:,2),Y);
hold on;
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
legend('-1','1','Support Vector');
m = - HardMarginSVM.Beta(1)/HardMarginSVM.Beta(2);
b = - HardMarginSVM.Bias/HardMarginSVM.Beta(2);
refline(m,b);
margin = 1/HardMarginSVM.Beta(2);
l1 = refline(m,b+margin);
l2 = refline(m,b-margin);
l1.LineStyle = '--';
l2.LineStyle = '--';
hold off;

% nsv = size(sv,1);
% for i = 1:nsv
%     sv(i,:)
%     HardMarginSVM.Beta(1)*sv(i,1)+HardMarginSVM.Beta(2)*sv(i,2)+HardMarginSVM.Bias
% end
% HardMarginSVM.Beta
% HardMarginSVM.Bias
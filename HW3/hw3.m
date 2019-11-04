height=size(loadimage(1,1),1);
width=size(loadimage(1,1),2);

X=loadimg(3,10);

meanface = mean(X,2);
meanface_image = reshape(meanface,[height width]);

[M,U_d,S,Y]=PrincipalComponents(X,2,meanface);
eigenface_1=reshape(U_d(:,1),[height width]);
eigenface_2=reshape(U_d(:,2),[height width]);

S=S/3;

% figure;
% subplot(1,3,1);
% imshow(uint8(meanface_image));
% subplot(1,3,2);
% imshow(uint8(S(1,1)*eigenface_1));
% subplot(1,3,3);
% imshow(uint8(S(2,2)*eigenface_2));

figure;
for i = 1:2
    for j = -1:0.2:1
        if i== 1
            number = nearest((j+1)/0.2)+1;
        else
            number = nearest((j+1)/0.2)+12;
        end
        subplot(2,11,number);
        imshow(uint8(reshape(meanface+j*S(i,i)*U_d(:,i),[height width])));
    end
end


function X=loadimg(individual,n)
    for i = 1:n
        img=loadimage(individual,i);
        pixel_num = size(img,1)*size(img,2);
        X(:,i)=reshape(img,[pixel_num 1]);
    end
end

function [M,U_d,S,Y]=PrincipalComponents(X,d,mean)
    M = mean*ones(1,size(X,2));
    XX = X-M;
    [U,S,V]=svd(XX,'econ');
    U_d=U(:,1:d);
    Y=transpose(U_d)*XX;
end


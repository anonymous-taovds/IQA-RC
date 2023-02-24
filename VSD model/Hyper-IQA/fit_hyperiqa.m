clc;
clear all;
close all;
% syms a as real



data=load('wmse_gppcam.txt');
num = data(:,1);
s=data(:,2);
wmse1=data(:,3);
psnr1=data(:,4);

data=load('wmse_scam.txt');
wmse2=data(:,3);
psnr2=data(:,4);

data=load('wmse_gcam.txt');
wmse3=data(:,3);
psnr3=data(:,4);

data=load('wmse_gppcam_2.txt');
wmse4=data(:,3);
psnr4=data(:,4);

data=load('wmse_scam_2.txt');
wmse5=data(:,3);
psnr5=data(:,4);

data=load('wmse_gcam_2.txt');
wmse6=data(:,3);
psnr6=data(:,4);

data=load('wmse_gppcam_3.txt');
wmse7=data(:,3);
psnr7=data(:,4);

data=load('wmse_scam_3.txt');
wmse8=data(:,3);
psnr8=data(:,4);

data=load('wmse_gcam_3.txt');
wmse9=data(:,3);
psnr9=data(:,4);

data=load('wmse_gppcam_gcam.txt');
wmse10=data(:,3);
psnr10=data(:,4);

data=load('wmse_scam_gcam.txt');
wmse11=data(:,3);
psnr11=data(:,4);

data=load('wmse_scam_gppcam.txt');
wmse12=data(:,3);
psnr12=data(:,4);

data=load('wmse_scam_gppcam_gcam.txt');
wmse13=data(:,3);
psnr13=data(:,4);

 
%鎷熷悎
x1=wmse1;
x2=wmse2;
x3=wmse3;
x4=wmse4;
x5=wmse5;
x6=wmse6;
x7=wmse7;
x8=wmse8;
x9=wmse9;
x10=wmse10;
x11=wmse11;
x12=wmse12;
x13=wmse13;

Y=s;

% X=[x1 x2 x3 x4 x5 x6 x7 x8 x9];
X=[x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13];

n=length(x1);
R2_best=0 ;
rSquare0_best=0;
for i=80:150
    a0=i.*rand(1,15);i;
%     func1=@(a,X)((abs(a(1)).*X(:,1)+abs(a(2)).*X(:,4)+abs(a(3)).*X(:,7)+abs(a(4)).*X(:,2)+abs(a(5)).*X(:,5)+abs(a(6)).*X(:,8)+abs(a(7)).*X(:,3)+abs(a(8)).*X(:,6)+abs(a(9)).*X(:,9))./(abs(a(1))+abs(a(2))+abs(a(3))+abs(a(4))+abs(a(5))+abs(a(6))+abs(a(7))+abs(a(8))+abs(a(9)))./abs(a(10))).^(-1/a(11));
%     func1=@(a,X)((abs(a(1)).*X(:,1)+abs(a(2)).*X(:,4)+abs(a(3)).*X(:,7)+abs(a(4)).*X(:,2)+abs(a(5)).*X(:,5)+abs(a(6)).*X(:,8)+abs(a(7)).*X(:,3)+abs(a(8)).*X(:,6)+abs(a(9)).*X(:,9))./(abs(a(1))+abs(a(4))+abs(a(7)))./abs(a(10))).^(-1/a(11));
%     func1=@(a,X)((abs(a(1)).*X(:,1)+abs(a(2)).*X(:,4)+abs(a(3)).*X(:,7)+abs(a(4)).*X(:,2)+abs(a(5)).*X(:,5)+abs(a(6)).*X(:,8)+abs(a(7)).*X(:,3)+abs(a(8)).*X(:,6)+abs(a(9)).*X(:,9)+abs(a(10)).*X(:,10)+abs(a(11)).*X(:,11)+abs(a(12)).*X(:,12)+abs(a(13)).*X(:,13))./(abs(a(1))+abs(a(4))+abs(a(7)))./abs(a(14))).^(-1/a(15));
%     func1=@(a,X)((abs(a(1)).*X(:,1)+abs(a(2)).*X(:,4)+abs(a(3)).*X(:,7)+abs(a(4)).*X(:,2)+abs(a(5)).*X(:,5)+abs(a(6)).*X(:,8)+abs(a(7)).*X(:,3)+abs(a(8)).*X(:,6)+abs(a(9)).*X(:,9)+abs(a(10)).*X(:,10)+abs(a(11)).*X(:,11)+abs(a(12)).*X(:,12)+abs(a(13)).*X(:,13))./(abs(a(1))+abs(a(2))+abs(a(3))+abs(a(4))+abs(a(5))+abs(a(6))+abs(a(7))+abs(a(8))+abs(a(9))+abs(a(10))+abs(a(11))+abs(a(12))+abs(a(13)))./abs(a(14))).^(-1/a(15));
%     func1=@(a,X)((abs(a(1)).*X(:,1)+abs(a(2)).*X(:,4)+abs(a(4)).*X(:,2)+abs(a(5)).*X(:,5)+abs(a(7)).*X(:,3)+abs(a(8)).*X(:,6)+abs(a(10)).*X(:,10)+abs(a(11)).*X(:,11)+abs(a(12)).*X(:,12))./(abs(a(1))+abs(a(2))+abs(a(4))+abs(a(5))+abs(a(7))+abs(a(8))+abs(a(10))+abs(a(11))+abs(a(12)))./abs(a(14))).^(-1/a(15));
%     func1=@(a,X)((abs(a(1)).*X(:,1)+abs(a(2)).*X(:,4)+abs(a(4)).*X(:,2)+abs(a(5)).*X(:,5)+abs(a(7)).*X(:,3)+abs(a(8)).*X(:,6))./(abs(a(1))+abs(a(2))+abs(a(4))+abs(a(5))+abs(a(7))+abs(a(8)))./abs(a(14))).^(-1/a(15));
    func1=@(a,X)((abs(a(1)).*X(:,1)+abs(a(4)).*X(:,2)+abs(a(7)).*X(:,3))./(abs(a(1))+abs(a(4))+abs(a(7)))./abs(a(14))).^(-1/a(15));





%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,1).^2+a(3).*X(:,1).^3+a(4).*X(:,2)+a(5).*X(:,2).^2+a(6).*X(:,2).^3)./(a(1)+a(2)+a(3)+a(4)+a(5)+a(6))./a(7))).^(-1/a(8));
%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,1).^2+a(3).*X(:,1).^3+a(4).*X(:,2)+a(5).*X(:,2).^2+a(6).*X(:,2).^3)./(a(1)+a(4))./a(7))).^(-1/a(8));
%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,1).^2+a(3).*X(:,1).^3+a(4).*X(:,3)+a(5).*X(:,3).^2+a(6).*X(:,2).^3)./(a(1)+a(2)+a(3)+a(4)+a(5)+a(6))./a(7))).^(-1/a(8));
%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,1).^2+a(3).*X(:,1).^3+a(4).*X(:,3)+a(5).*X(:,3).^2+a(6).*X(:,3).^3)./(a(1)+a(4))./a(7))).^(-1/a(8));
%     func1=@(a,X)(abs((a(1).*X(:,2)+a(2).*X(:,2).^2+a(3).*X(:,2).^3+a(4).*X(:,3)+a(5).*X(:,3).^2+a(6).*X(:,2).^3)./(a(1)+a(2)+a(3)+a(4)+a(5)+a(6))./a(7))).^(-1/a(8));
%     func1=@(a,X)(abs((a(1).*X(:,2)+a(2).*X(:,2).^2+a(3).*X(:,2).^3+a(4).*X(:,3)+a(5).*X(:,3).^2+a(6).*X(:,3).^3)./(a(1)+a(4))./a(7))).^(-1/a(8));
%     func1=@(a,X)(abs((a(1).*X(:,1).*X(:,2)+a(2).*X(:,2).*X(:,3)+a(3).*X(:,1).*X(:,3))./(a(1)+a(2)+a(3))./a(4))).^(-1/a(5));
%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,1).^2+a(3).*X(:,1).^3)./a(7))).^(-1/a(8))+abs(((a(4).*X(:,2)+a(5).*X(:,2).^2+a(6).*X(:,2).^3)./a(9))).^(-1/a(10))+abs(((a(11).*X(:,2)+a(12).*X(:,2).^2+a(13).*X(:,2).^3)./a(14))).^(-1/a(15));
%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,1).^2+a(3).*X(:,1).^3+a(4).*X(:,2)+a(5).*X(:,2).^2+a(6).*X(:,2).^3+a(7).*X(:,3)+a(8).*X(:,3).^2+a(9).*X(:,3).^3)./(a(1)+a(2)+a(3)+a(4)+a(5)+a(6)+a(7)+a(8)+a(9))./a(10))).^(-1/a(11))+(abs((a(12).*X(:,1)+a(13).*X(:,1).^2+a(14).*X(:,1).^3+a(15).*X(:,2)+a(16).*X(:,2).^2+a(17).*X(:,2).^3+a(18).*X(:,3)+a(19).*X(:,3).^2+a(20).*X(:,3).^3)./(a(12)+a(13)+a(14)+a(15)+a(16)+a(17)+a(18)+a(19)+a(20))./a(21))).^(-1/a(22));
%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,1).^2+a(3).*X(:,1).^3+a(4).*X(:,2)+a(5).*X(:,2).^2+a(6).*X(:,2).^3+a(7).*X(:,3)+a(8).*X(:,3).^2+a(9).*X(:,3).^3)./(a(1)+a(4)+a(7))./a(10))).^(-1/a(11))+(abs((a(12).*X(:,1)+a(13).*X(:,1).^2+a(14).*X(:,1).^3+a(15).*X(:,2)+a(16).*X(:,2).^2+a(17).*X(:,2).^3+a(18).*X(:,3)+a(19).*X(:,3).^2+a(20).*X(:,3).^3)./(a(12)+a(15)+a(18))./a(21))).^(-1/a(22));
%     func1=@(a,X)(abs((a(1).*X(:,1)+a(2).*X(:,2)+a(3).*X(:,3))./(a(1)+a(2)+a(3))./a(4))).^(-1/a(5));

%     [a,r,J] = nlinfit(X,Y,func1,a0);a;
%     a=real(a); a;
    rSquare_temp=zeros(1,24);
    plcc_temp=zeros(1,24);
    a=zeros(15,24);
    Y0=zeros(240,1);
    for j=1:24
        [a1,r,J] = nlinfit(X(j:24:(216+j),:),Y(j:24:(216+j),:),func1,a0);
        a(:,j)=a1;
        Y1=func1(a1,X(j:24:(216+j),:));
        meany = mean(Y(j:24:(216+j)));
        SST = sum((Y(j:24:(216+j)) - meany).^2);
        SSE = sum((Y(j:24:(216+j)) - Y1).^2);
        Y0(j:24:(216+j))=Y1;
        rSquare_temp(j) = 1-SSE/SST;
        plcc_temp(j) = corr(Y(j:24:(216+j)),Y1,'type','Pearson');
    end
    rSquare=mean(rSquare_temp);
    plcc=mean(plcc_temp);

    meany = mean(Y);
    SST = sum((Y - meany).^2);
    SSE = sum((Y - Y0).^2);
    rSquare0 = 1-SSE/SST;
    plcc0 = corr(Y,Y0,'type','Pearson');

    if rSquare<1 && rSquare>R2_best
        R2_best=rSquare;
        plcc_best=plcc;
        rSquare0_best=rSquare0;
        plcc0_best=plcc0;
        plcc_temp_best=plcc_temp;
        rSquare_temp_best=rSquare_temp;
        a_best=a;
    end
end
                                         


a_best
R2_best
rSquare_temp_best
plcc_best
plcc_temp_best


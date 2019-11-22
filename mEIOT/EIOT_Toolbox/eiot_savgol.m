function [dmdx,M]=eiot_savgol(dm,ws,op,od)
isrow=0;
if (size(dm,1) ==1 && size(dm,2)>1)
    l = size(dm,2);
    dm = dm';
    isrow=1;
elseif  (size(dm,1)>1 && size(dm,2)==1)
    l = size(dm,1);
else
    l = size(dm,2);
end
x_vec=[-ws:ws]';
X=[ones(2*ws+1,1)];
for oo=1:op
    X=[X,x_vec.^oo];
end

XtXiXt=inv(X' * X) * X';
coeffs=XtXiXt(od+1,:) * factorial(od);
M=[];
for i=[1:l-2*ws]
    m_=[zeros(1,i-1),coeffs,zeros(1,l-2*ws-i)];
    M=[M;m_] ;
end

if isvector(dm)
    dmdx=M*dm;
    if isrow 
        dmdx=dmdx';
    end
else
    dmdx=[];
    for i=1:size(dm,1)
        dmdx_=M*dm(i,:)';
        dmdx=[dmdx;dmdx_'];
    end
end
%
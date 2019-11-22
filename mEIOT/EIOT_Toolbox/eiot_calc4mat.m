function [r_hat,ri_hat,ssr] = eiot_calc4mat(Dm,eiot_obj,varargin)
%        [C_E_hat,SSE] = eiot4mat(Dm,eiot_obj,<sum_r_nrs,rk>)
rk=NaN;
rk_given  =0;
sum_r_nrs =0;
r_hat=[];
ri_hat=[];
ssr=[];
if nargin ==3
  sum_r_nrs=varargin{1};  
end
if nargin==4
   sum_r_nrs=varargin{1}; 
   rk=varargin{2};    
   rk_given=1;
end
if isscalar(sum_r_nrs) 
sum_r_nrs=repmat(sum_r_nrs,size(Dm,1),1);
end

if isscalar(rk)
    if isnan(rk)
        rk_given=0;
    end
end

for i=1:size(Dm,1)
    if rk_given    
     [r_hat_,ri_hat_,ssr_] = eiot_calc(Dm(i,:),eiot_obj,sum_r_nrs(i),rk(i,:));
    else
     [r_hat_,ri_hat_,ssr_] = eiot_calc(Dm(i,:),eiot_obj,sum_r_nrs(i));   
    end
    r_hat  = [r_hat;r_hat_'];
    ri_hat = [ri_hat;ri_hat_'];
    ssr     = [ssr ; ssr_];
end

end


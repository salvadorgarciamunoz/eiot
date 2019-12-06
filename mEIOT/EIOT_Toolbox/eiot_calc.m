function [r_hat,ri_hat,ssr,m] = eiot_calc(Dm,eiot_obj,varargin)
%        [r_hat,ri_hat,ssr,m] = eiot_calc(Dm,eiot_obj,<sum_r_nrs,rk>)
%Inputs:
% Dm        : Matrix of measure spectra (o x l)
%
% eiot_obj  : Structure given by eiot_build.m
%             
%
% Optional:
% sum_r_nrs : Summation of mass fractions for non-resolved species (0 if
%             ommited) - scalar if same for all observations 
%                        vector (o x 1 ) if not.
%
% rk        : Vector of known non-chemical interfernces (o x m)
%
%Outputs:
% r_hat  : Fractions per specie
% ri_hat : Strength of non-chemical interferences
% ssr    : Squared Spectral Residual, sum of squares of residuals after EIOT
%          deflation.
% m      : Mahalanobis distance of the ri_hat

    if nargin ==3
      sum_r_nrs = varargin{1};  
      rk        = [];
    elseif nargin==4
       sum_r_nrs=varargin{1}; 
       rk=varargin{2};    
    else
       sum_r_nrs =0;  
       rk        = [];
    end
    if isscalar(sum_r_nrs) 
        sum_r_nrs=repmat(sum_r_nrs,size(Dm,1),1);
    end

    if iscolumn(Dm) 
        Dm=Dm';
    end

    r_hat  = [];
    ri_hat = [];
    ssr    = [];
    m      = [];

    for i=1:size(Dm,1)
        if ~isempty(rk)    
         [r_hat_,ri_hat_,ssr_,m_] = eiot_calc_vec(Dm(i,:),eiot_obj,sum_r_nrs(i),rk(i,:));
        else
         [r_hat_,ri_hat_,ssr_,m_] = eiot_calc_vec(Dm(i,:),eiot_obj,sum_r_nrs(i));   
        end
        r_hat  = [r_hat  ; r_hat_' ];
        ri_hat = [ri_hat ; ri_hat_'];
        ssr    = [ssr    ; ssr_    ];
        m      = [m      ; m_      ];    
    end

end

function [r_hat,ri_hat,ssr,m] = eiot_calc_vec(dm,eiot_obj,varargin)
%        [r_hat,ri_hat,ssr,m] = eiot_calc_vec(dm,eiot_obj,<sum_r_nrs,rk>)
%Inputs:
% dm        : Vector of measure spectra 
%
% eiot_obj  : Structure given by eiot_build.m
%             
%
% Optional:
% sum_r_nrs : Summation of mass fraction for non-resolved species (0 if
%             ommited)
% rk        : Vector of known non-chemical interfernces
%
%Outputs:
% r_hat  : Fractions per specie
% ri_hat : Strength of non-chemical interferences
% ssr    : Squared Spectral Residual, sum of squares of residuals after EIOT
%          deflation.
% m      : Mahalanobis distance of the ri_hat


if nargin ==3
    sum_r_nrs = varargin{1};
    rk        = [];
elseif nargin ==4
    sum_r_nrs = varargin{1};
    rk        = varargin{2};
else
    sum_r_nrs =  0;
    rk        = [];
end


if isrow(dm)
    dm=dm';   % Assure dm has the correct orientation [ column vector ]
end

if ~isempty(rk)
    if isrow(rk)
      rk=rk';   % Assure rk has the correct orientation [ column vector ]
    end
    rk = rk -  eiot_obj.rk_stats.mean';
    rk = rk ./ eiot_obj.rk_stats.std';
end
    
if eiot_obj.num_e_si == 0 
    H   =  eiot_obj.S_E*eiot_obj.S_E';
    f   = -(dm'*eiot_obj.S_E');
    A   = [];
    b   = [];
    Aeq = [ones(1,size(eiot_obj.S_E,1)-eiot_obj.num_si) zeros(1,eiot_obj.num_si)];
%    Aeq = [ones(1,size(eiot_obj.S_hat,1)) zeros(1,eiot_obj.num_sI)];                     
    beq = 1-sum_r_nrs;
    if ~isempty(rk)
        Aeq_ = diag(eiot_obj.index_rk_eq);
        Aeq_ = Aeq_(sum(Aeq_,2)==1,:);
        beq_ = rk;
        Aeq = [Aeq; Aeq_];
        beq = [beq; beq_];
    end
    c_E_hat = quadprog(H,f,A,b,Aeq,beq);
    r_hat   = c_E_hat(1:size(eiot_obj.S_hat,1));
    ri_hat  = c_E_hat(size(eiot_obj.S_hat,1)+1:end);
    dm_hat  = eiot_obj.S_E'*c_E_hat;
    ssr     = sum((dm - dm_hat).^2);
    m       = sum((ri_hat.^2)./repmat(var(ri_hat),size(ri_hat,1),1),1);
else 
    me     = eiot_obj.num_e_si;
    S_I_E =  eiot_obj.S_E(end-me+1:end,:);
    SE_aux = eiot_obj.S_E(1:end-me,:); 
    SSR_m = [];
    
    A       =  [];
    b       =  [];
    
    for m = 1:me
        % Evaluate one Exclusive at a time
        
        %Prepare quadratic programming problem
        SE_m      = [SE_aux;S_I_E(m,:)];
        H         =  SE_m*SE_m';
        f         = -(dm'*SE_m');
        
        %add sumation to unity constraint
        Aeq       = [ones(1,size(SE_m,1)-eiot_obj.num_si+me-1) zeros(1,eiot_obj.num_si-me+1)];
        beq       = 1-sum_r_nrs;
        
        %add forced binary as a constraint
        Aeq_      = zeros(1,size(SE_m,1));
        Aeq_(end) = 1;
        beq_      = 1;
        Aeq       = [Aeq; Aeq_];
        beq       = [beq; beq_];
        
        %If applicable add the equalities for rk
        if ~isempty(rk)
            Aeq_ = diag(eiot_obj.index_rk_eq);
            Aeq_ = Aeq_(sum(Aeq_,2)==1,:);
            beq_ = rk;
            Aeq = [Aeq; Aeq_];
            beq = [beq; beq_];
        end
        
        %Solve
        c_E_hat = quadprog(H,f,A,b,Aeq,beq);
        dm_hat  = SE_m'*c_E_hat;
        ssr_    = sum((dm - dm_hat).^2);
        SSR_m   = [SSR_m ; ssr_];
    end
    
    %find scenario with least ssr
    [~,indx] = min(SSR_m);
    
    %Prepare quadratic programming problem with best case
    SE_m    = [SE_aux;S_I_E(indx,:)];
    H       =  SE_m*SE_m';
    f       = -(dm'*SE_m');
    
    %add sumation to unity constraint
    Aeq     = [ones(1,size(SE_m,1)-eiot_obj.num_si+me-1) zeros(1,eiot_obj.num_si-me+1)];
    beq     = 1-sum_r_nrs;
    
    %add forced binary = 1 as a constraint
    Aeq_      = zeros(1,size(SE_m,1));
    Aeq_(end) = 1;
    beq_      = 1;
    Aeq       = [Aeq; Aeq_];
    beq       = [beq; beq_];
    
    %If applicable add equality constratints for rk
    if ~isempty(rk)
        Aeq_ = diag(eiot_obj.index_rk_eq);
        Aeq_ = Aeq_(sum(Aeq_,2)==1,:);
        beq_ = rk;
        Aeq = [Aeq; Aeq_];
        beq = [beq; beq_];
    end
 
    %solve
    c_E_hat = quadprog(H,f,A,b,Aeq,beq);
    
    dm_hat  = SE_m'*c_E_hat;
    ssr     = sum((dm - dm_hat).^2);
    
    %rebuild exclusive part of ri 
    rie       = zeros(me,1);
    rie(indx) = 1;  
    c_E_hat = [c_E_hat(1:end-me);rie]; 
    
    r_hat   = c_E_hat(1:size(eiot_obj.S_hat,1));
    ri_hat  = c_E_hat(size(eiot_obj.S_hat,1)+1:end);
    m       = sum((ri_hat.^2)./repmat(var(ri_hat),size(ri_hat,1),1),1);
    
end % close end to main IF loop

end




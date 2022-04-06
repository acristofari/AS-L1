% -------------------------------------------------------------------------
% 
% This file is part of AS-L1, which is a solver for the following problems
% with l1 constraint:
% 
%                                 min f(x)
%                           s.t. ||x||_1 <= tau,
% 
% with given objective function f(x) and l1-ball radius tau.
% 
% In this file, we consider logistic regression problems, that is,
% 
%               f(x) = sum_{i=1}^m log(1+exp(-y_i A_i^T x)),
% 
% with given matrix A = [A_1 ... A_m]^T and vector y = [y_1 ... y_m].
% 
% -------------------------------------------------------------------------
% 
% Reference paper:
% 
% A. Cristofari, M. De Santis, S. Lucidi, F. Rinaldi (2022). Minimization
% over the l1-ball using an active-set non-monotone projected gradient. 
% arXiv preprint 2108.00237.
% 
% -------------------------------------------------------------------------
% 
% Authors:
% Andrea Cristofari (e-mail: andrea.cristofari@unipd.it)
% Marianna De Santis (e-mail: mdesantis@diag.uniroma1.it)
% Stefano Lucidi (e-mail: lucidi@diag.uniroma1.it)
% Francesco Rinaldi (e-mail: rinaldi@math.unipd.it)
% 
% Last update of this file:
% April 6th, 2022
% 
% Licensing:
% This file is part of AS-L1.
% AS-L1 is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% AS-L1 is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with AS-L1. If not, see <http://www.gnu.org/licenses/>.
% 
% Copyright 2022 Andrea Cristofari, Marianna De Santis, Stefano Lucidi,
% Francesco Rinaldi.
% 
% -------------------------------------------------------------------------

function [x,f,as_l1_info] = as_l1_log_reg(A,y,tau,opts)
    
    if (nargin < 3)
        error('at least three input arguments are required');
    end
    if (nargin > 4)
        error('at most four input arguments are required');
    end
    if (nargout > 3)
        error('at most three output arguments are required');
    end
    
    if (~isnumeric(A) || ~isreal(A) || ~ismatrix(A))
        error('the first input must be a real matrix');
    end
    if (~isnumeric(y) || ~isreal(y) || ~iscolumn(y))
        error('the second input must be a real column vector');
    end
    if (size(A,1) ~= length(y))
        error('the number of rows of the first input must be equal to the length of the second input');
    end
    if (~isnumeric(tau) || ~isreal(tau) || ~isscalar(tau) || tau<=0e0)
        error('the third input must be a positive number');
    end
    
    n = size(A,2);
    
    % set options
    eps_opt = 1e-5;
    m = 10;
    max_it = 10000;
    min_f = -Inf;
    min_gd = 1e-20;
    min_stepsize = 1e-12;
    proj_l1ball = @(x) max(abs(x)-max(max((cumsum(sort(abs(x),1,'descend'),1)-tau)./(1:size(x,1))'),0),0).*sign(x); % from https://lcondat.github.io/software
    x = zeros(n,1);
    verbosity = false;
    if (nargin == 4)
        if (~isstruct(opts) || ~isscalar(opts))
            error('the fourth input (which is optional) must be a structure');
        end
        opts_field = fieldnames(opts);
        for i = 1:length(opts_field)
            switch char(opts_field(i))
                case 'eps_opt'
                    eps_opt = opts.eps_opt;
                    if (~isnumeric(eps_opt) || ~isreal(eps_opt) || ~isscalar(eps_opt) || eps_opt<0e0)
                       error('in the options, ''eps_opt'' must be a non-negative real number');
                    end
                case 'ls_memory'
                    m = opts.ls_memory;
                    if (~isnumeric(m) || ~isreal(m) || ~isscalar(m) || m<1e0)
                       error('in the options, ''linesearch_memory'' must be a real number greater than or equal to 1');
                    end
                    m = floor(m);
                case 'max_it'
                    max_it = opts.max_it;
                    if (~isnumeric(max_it) || ~isreal(max_it) || ~isscalar(max_it) || max_it<1e0)
                       error('in the options, ''max_it'' must be a real number greater than or equal to 1');
                    end
                    max_it = floor(max_it);
                case 'min_f'
                    min_f = opts.min_f;
                    if (~isnumeric(min_f) || ~isreal(min_f) || ~isscalar(min_f))
                       error('in the options, ''f_stop'' must be a real number');
                    end
                case 'min_gd'
                    min_gd = opts.min_gd;
                    if (~isnumeric(min_gd) || ~isreal(min_gd) || ~isscalar(min_gd) || min_gd<0e0)
                       error('in the options, ''min_gd'' must be greater than or equal to 0');
                    end
                case 'min_stepsize'
                    min_stepsize = opts.min_stepsize;
                    if (~isnumeric(min_gd) || ~isreal(min_gd) || ~isscalar(min_gd) || min_gd<0e0)
                       error('in the options, ''min_stepsize'' must be greater than or equal to 0');
                    end
                case 'proj'
                    proj_l1ball = opts.proj;
                    if (~isa(proj_l1ball,'function_handle'))
                      error('in the options, ''proj'' must be a function handle.');
                   end
                case 'x0'
                    x = opts.x0;
                    if (~isnumeric(x) || ~isreal(x) || ~iscolumn(x) || length(x) ~= n)
                        error('in the options, ''x0'' must be a real column vector with length equal to the number of columns of A');
                    end
                case 'verbosity'
                    verbosity = opts.verbosity;
                    if (~islogical(verbosity) || ~isscalar(verbosity))
                       error('in the options, ''verbosity'' must be a logical');
                    end
                otherwise
                    error('not valid field name in the structure of options');
            end
        end
    end
    
    % direction parameters
    c_bb_min = 1e-10;
    c_bb_max = 1e10;
    
    % line search parameters
    gamma = 1e-3;
    delta = 5e-1;
    
    if (norm(x,1) > tau)
        x = proj_l1ball(x);
    end
    
    nnz_max = round(n/3);
    
    it = 0;
    
    if (sum(x~=0e0) < nnz_max)
        Ax = A*sparse(x);
    else
        Ax = A*x;
    end
    v = exp(y.*Ax);
    f = sum(log((1e0./v)+1e0));
    
    if (f <= min_f)
        flag = 4;
        as_l1_info.flag = flag;
        as_l1_info.it = it;
        return;
    end
    
    flag = 0;
    act = false(n,1);
    act_old = false(n,1);
    
    eps_as = 1e-6;
    
    w = f*ones(m,1);
    	
    if (verbosity)
        fprintf('%s%.4e\n','it = 0, f = ',f);
    end
	
    while (true)
        
        g = -A'*(y./(v+1e0));
        
        % calculate multipliers
        lambda = g'*x;
        tau_g = tau*g;
        
        new_point_act_phase = false;
        
        % -----------------------------------------------------------
        %    minimization step over the estimated active variables
        % -----------------------------------------------------------
        while (~new_point_act_phase)
            
            act = ( x>=0 & x<=eps_as*tau*(tau_g-lambda) & eps_as*tau*(tau_g+lambda)<=0 ) | ...
                  ( x<=0 & x>=eps_as*tau*(tau_g+lambda) & eps_as*tau*(tau_g-lambda)>=0 );
            non_act = find(~act);
            ax = and(act,not(act_old));
            
            if (any(x(ax)~=0))
                
                % compute the index j
                [~,j] = max(abs(g(non_act)));
                
                % fix the estimated active variables to zero and move the j-th variable
                x_tilde = x;
                x_tilde(non_act(j)) = x(non_act(j)) - sign(g(non_act(j)))*sum(abs(x(ax)));
                x_tilde(ax) = 0e0;
                
                % compute the objective function at the new point
                if (sum(x~=0e0) < nnz_max)
                    Ax_tilde = A*sparse(x_tilde);
                elseif (sum(ax) < nnz_max)
                    d = zeros(n,1);
                    d(ax) = -x(ax);
                    d(non_act(j)) = x_tilde(non_act(j)) - x(non_act(j));
                    Ax_tilde = Ax + A*sparse(d);
                else
                    Ax_tilde = A*x_tilde;
                end
                v_tilde = exp(y.*Ax_tilde);                
                f_tilde = sum(log((1e0./v_tilde)+1e0));
                
                % check if a significant decrease in the objective function is obtained
                if (f_tilde-f <= -1e-6*((x-x_tilde)'*(x-x_tilde))/eps_as)
                    new_point_act_phase = true;
                    x = x_tilde;
                    f = f_tilde;
                    g = -A'*(y./(v_tilde+1e0));
                    Ax = Ax_tilde;
                    v = v_tilde;
                else
                    eps_as = 1e-1*eps_as;
                end
                
            else
                break;
            end
            
        end
        
        % -----------------------------------------------------------
        %  minimization step over the estimated non-active variables
        % -----------------------------------------------------------
        
        % compute direction
        if (it >= 1)
            s = x(non_act) - x_old(non_act);
            y = g(non_act) - g_old(non_act);
            sy = s'*y;
            if (sy > 0e0)
                c_bb = (s'*s)/sy;
                if (c_bb < c_bb_min)
                    c_bb = sy/(y'*y);
                    if (c_bb < c_bb_min)
                        c_bb = c_bb_min;
                    else
                        c_bb = min(c_bb,c_bb_max);
                    end
                else
                    c_bb = min(c_bb,c_bb_max);
                end
            else
                c_bb = min(c_bb_max,max(1e0,norm(x(non_act))/norm(g(non_act))));
            end
            xg = x - c_bb*g;
        else
            xg = x - g;
        end
        
        xg_proj_nonact = proj_l1ball(xg(non_act));
        d = zeros(n,1);
        d(non_act) = xg_proj_nonact - x(non_act);
        sup_norm_d_non_act = norm(d(non_act),Inf);
        
        % test for termination
        if (norm(x-proj_l1ball(x-g),Inf) <= eps_opt)
            if (new_point_act_phase)
                it = it + 1;
                if (verbosity)
                    fprintf('%s%i%s%.4e\n','it = ',it,', f = ',f);
                end
            end
            break;
        end
        
        if (sup_norm_d_non_act > 0e0)
            gd = g'*d;
            if (gd < -min_gd)
                w = [f; w(1:m-1)];
                f_ref = max(w);
                gamma_gd = gamma*gd;
                if (it >= 1)
                    alpha = 1e0;
                    z = x + d;
                else
                    alpha = min(1e0,1e0/norm(g(non_act),2));
                    z = x + alpha*d;
                end
                nnz_d = sum(d~=0e0);
                while (true)
                    if (nnz_d < nnz_max)
                        Az = Ax + alpha*(A*sparse(d));
                    else
                        Az = A*z;
                    end                      
                    vz = exp(y.*Az);
                    f = sum(log((1e0./vz)+1e0));
                    if (f <= f_ref + alpha*gamma_gd)
                        x_old = x;
                        x = z;
                        v = vz;
                        Ax = Az;
                        break;
                    else
                        alpha = delta*alpha;
                        if (alpha <= min_stepsize)
                            flag = 2;
                            break;
                        end
                        z = x + alpha*d;
                    end
                end
            else
                flag = 1;
            end
        else
            x_old = x;
        end
        
        if (flag > 0)
            if (~new_point_act_phase)
                break;
            else
                flag = 0;
            end
        end
        
        it = it + 1;
                
        if (verbosity)
            fprintf('%s%i%s%.4e\n','it = ',it,', f = ',f);
        end
        
        if (it >= max_it)
            flag = 3;
            break;
        end
        if (f <= min_f)
            flag = 4;
            break;
        end
        
        g_old = g;
        act_old = act;
        
    end
    
    as_l1_info.flag = flag;
    as_l1_info.it = it;
    
end
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
% In this file, we consider lasso problems, that is,
% 
%                           f(x) = 0.5||Ax-b||^2,
% 
% with given matrix A and vector b.
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
% April 7th, 2022
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


clear all, clc;

rng(1);

% In this file, it is shown how to call AS-L1 to solve a user-defined problem

% (1) Get the problem (i.e., matrix 'A', vector 'b' and l1-ball radius 'tau')
m = 2^9;
n = 2^10;
A = rand(m,n);
xstar = sign(sprandn(n,1,floor(0.05*m)/n));
b = A*xstar + 0.001*randn(m,1);
tau = 0.99*norm(xstar,1);

% (2) Call AS-L1
[x,f,as_l1_info] = as_l1_lasso(A,b,tau);

%--------------------------------------------------------------------------
%
% *** EXAMPLE OF HOW TO CHANGE AS-L1 PARAMETERS ***
%
% (see the file 'usage.txt' to know which parameters can be changed and
% their default values)
%
% Instead of calling AS-L1 by the above instruction, do the following:
%
% - create a structure having as field names the names of the parameters
%   to be changed and assign them new values, e.g.,
%
%     opts.verbosity = true;
%
% - pass the structure to AS-L1 as fourth input argument, e.g.,
%
%     [x,f,as_l1_info] = as_l1_lasso(A,b,tau,opts);
%
%--------------------------------------------------------------------------

fprintf(['************************************************' ...
         '\nAlgorithm: AS-L1' ...
         '\nf = %-.5e' ...
         '\nnumber of iterations = %-i' ...
         '\nexit flag = %-i' ...
         '\n************************************************\n'], ...
        f,as_l1_info.it,as_l1_info.flag);
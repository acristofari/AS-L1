# Minimization over the ℓ1-ball

_Active-Set algorithm for minimization over the ℓ1-ball_ (AS-L1) is a solver for the following problem with ℓ1 constraint:

          min f(x)
    s.t. ||x||_1 <= τ,

with given objective function _f(x)_ and _ℓ1_-ball radius _τ._

AS-L1 uses an active-set strategy and a projected spectral gradient direction with non-monotone line search.

## Reference paper

[A. Cristofari, M. De Santis, S. Lucidi, F. Rinaldi (2022). _Minimization over the l1-ball using an active-set non-monotone projected gradient._
arXiv preprint 2108.00237.](https://arxiv.org/abs/2108.00237)

## Authors

* Andrea Cristofari (e-mail: [andrea.cristofari@unipd.it](mailto:andrea.cristofari@unipd.it))
* Marianna De Santis (e-mail: [mdesantis@diag.uniroma1.it](mailto:mdesantis@diag.uniroma1.it))
* Stefano Lucidi (e-mail: [lucidi@diag.uniroma1.it](mailto:lucidi@diag.uniroma1.it))
* Francesco Rinaldi (e-mail: [rinaldi@math.unipd.it](mailto:rinaldi@math.unipd.it))

## Licensing

AS-L1 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
AS-L1 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with AS-L1. If not, see <http://www.gnu.org/licenses/>.

Copyright 2022 Andrea Cristofari, Marianna De Santis, Stefano Lucidi, Francesco Rinaldi.

## How to use AS-L1

This directory should contain the files `COPYING.txt` and `README.md`,
plus the following subdirectories:

* `lasso`, with a Matlab implementation of AS-L1 to solve lasso problems. Namely, the objective function has the form

        f(x) = 0.5||Ax-b||^2,

   with given matrix _A_ and vector _b._

   This subdirectory should contain the following files:

     - `as_l1_lasso.m`, where the algorithm is implemented;
     - `main.m`, with an example of how to call AS-L1 in Matlab;
     - `usage.txt`, where inputs, outputs and options of the algorithm are explained in detail.

* `logistic_regression`, with a Matlab implementation of AS-L1 to solve logistic regression problems. Namely, the objective function has the form

        f(x) = sum_{i=1}^m log(1+exp(-y_i A_i^T x)),

   with given matrix _A = [A\_1 ... A\_m]^T_ and vector _y = [y\_1 ... y\_m]._

   This subdirectory should contain the following files:

     - `as_l1_log_reg.m`, where the algorithm is implemented;
     - `main.m`, with an example of how to call AS-L1 in Matlab;
     - `usage.txt`, where inputs, outputs and options of the algorithm are explained in detail.
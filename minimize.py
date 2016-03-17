# Minimize a continuous differentialble multivariate function. Starting point
# is given by "X" (D by 1), and the function named in the string "f", must
# return a function value and a vector of partial derivatives. The Polack-
# Ribiere flavour of conjugate gradients is used to compute search directions,
# and a line search using quadratic and cubic polynomial approximations and the
# Wolfe-Powell stopping criteria is used together with the slope ratio method
# for guessing initial step sizes. Additionally a bunch of checks are made to
# make sure that exploration is taking place and that extrapolation will not
# be unboundedly large. The "length" gives the length of the run: if it is
# positive, it gives the maximum number of line searches, if negative its
# absolute gives the maximum allowed number of function evaluations. You can
# (optionally) give "length" a second component, which will indicate the
# reduction in function value to be expected in the first line-search (defaults
# to 1.0). The function returns when either its length is up, or if no further
# progress can be made (ie, we are at a minimum, or so close that due to
# numerical problems, we cannot get any closer). If the function terminates
# within a few iterations, it could be an indication that the function value
# and derivatives are not consistent (ie, there may be a bug in the
# implementation of your "f" function). The function returns the found
# solution "X", a vector of function values "fX" indicating the progress made
# and "i" the number of iterations (line searches or function evaluations,
# depending on the sign of "length") used.
#
# Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, P4, P5)
#
# See also: checkgrad
#
# Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
#
#
# (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
#
# Permission is granted for anyone to copy, use, or modify these
# programs and accompanying documents for purposes of research or
# education, provided this copyright notice is retained, and note is
# made of any changes that have been made.
#
# These programs and documents are distributed without any warranty,
# express or implied.  As the programs were written for research
# purposes only, they have not been tested to the degree that would be
# advisable in any important application.  All use of these programs is
# entirely at the user's own risk.
#
#
# [Yue Cao (antares_tsao@hotmail.com)] Changes Made:
# 0) This is the "fmincg" function that was used in Andrew Ng's Machine Learning open course.
#    The original code is available at: http://www.gatsby.ucl.ac.uk/~edward/code/minimize/
# 1) translated original code into Python by Yue Cao, 2015
# 2) removed input arguments p1, p2, p3, p4, p5
# 3) add default length = 100

"""
this module contains a straight forward python translation of Carl Edward Rasmussen's original
U{minimize.m, http://www.gatsby.ucl.ac.uk/~edward/code/minimize/minimize.m}
"""

import numpy as np
from numpy.lib.scimath import sqrt


# # function [X, fX, i] = minimize(X, f, length, P1, P2, P3, P4, P5);
def minimize(f, X, length=100):
    """
    a straight forward python translation of Carl Edward Rasmussen's original
    U{minimize.m<http://www.gatsby.ucl.ac.uk/~edward/code/minimize/minimize.m>}

    each line starts with "# # " is the code from the original .m file

    @param f:
    @param X:
    @param length:
    @type length:
    @return:
    """
# # RHO = 0.01;                                    % a bunch of constants for line searches
    RHO = 0.01
# # SIG = 0.5;               % RHO and SIG are the constants in the Wolfe-Powell conditions
    SIG = 0.5
# # INT = 0.1;            % don't reevaluate within 0.1 of the limit of the current bracket
    INT = 0.1
# # EXT = 3.0;                            % extrapolate maximum 3 times the current bracket
    EXT = 3.0
# # MAX = 20;                                 % max 20 function evaluations per line search
    MAX = 20
# # RATIO = 100;                                              % maximum allowed slope ratio
    RATIO = 100
# #

# # argstr = [f, '(X'];                              % compose string used to call function
# # for i = 1:(nargin - 3)
# #   argstr = [argstr, ',P', int2str(i)];
# # end
# # argstr = [argstr, ')'];
    pass
# #
# # if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
    red = 1
    if (type(length) != int) and (len(length) == 2):
        red = length[1]
        length = length[0]

# # if length>0, S=['Linesearch']; else S=['Function evaluation']; end
    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'
# #
# # i = 0;                                                    % zero the run length counter
    i = 0
# # ls_failed = 0;                                     % no previous line search has failed
    ls_failed = 0
# # fX = [];
    fX = []
# # [f1 df1] = eval(argstr);                              % get function value and gradient
    f1, df1 = f(X)
# # i = i + (length<0);                                                    % count epochs?!
    i += (length < 0)
# # s = -df1;                                                % search direction is steepest
    s = -df1
# # d1 = -s'*s;                                                         % this is the slope
    d1 = -s.dot(s)
# # z1 = red/(1-d1);                                          % initial step is red/(|s|+1)
    z1 = red/(1-d1)
# #
# # while i < abs(length)                                              % while not finished
    while i < abs(length):
# #     i = i + (length>0);                                            % count iterations?!
        i += (length > 0)
# #
# #     X0 = X; f0 = f1; df0 = df1;                         % make a copy of current values
        X0 = X
        f0 = f1
        df0 = df1
# #     X = X + z1*s;                                                   % begin line search
        X = (X + (z1 * s))
# #     [f2 df2] = eval(argstr);
        f2, df2 = f(X)
# #     i = i + (length<0);                                                % count epochs?!
        i += (length < 0)
# #     d2 = df2'*s;
        d2 = df2.dot(s)
# #     f3 = f1; d3 = d1; z3 = -z1;                   % initialize point 3 equal to point 1
        f3 = f1
        d3 = d1
        z3 = -z1
# #     if length>0, M = MAX; else M = min(MAX, -length-i); end
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length-i)
# #     success = 0; limit = -1;                                    % initialize quanteties
        success = 0
        limit = -1
# #     while 1
        while True:
# #         while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0)
            while ((f2 > f1+z1*RHO*d1) or (d2 > -SIG*d1)) and (M > 0):
# #             limit = z1;                                           % tighten the bracket
                limit = z1
# #             if f2 > f1
                if f2 > f1:
# #                 z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
                    z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)
# #             else
                else:
# #                 A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
                    A = 6*(f2-f3)/z3+3*(d2+d3)
# #                 B = 3*(f3-f2)-z3*(d3+2*d2);
                    B = 3*(f3-f2)-z3*(d3+2*d2)
# #                 z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
                    z2 = (np.sqrt(B*B-A*d2*z3*z3)-B)/A
# #             end

# #             if isnan(z2) | isinf(z2)
                if np.isnan(z2) or np.isinf(z2):
# #                 z2 = z3/2;                  % if we had a numerical problem then bisect
                    z2 = z3/2
# #             end

# #             z2 = max(min(z2, INT*z3),(1-INT)*z3);    % don't accept too close to limits
                z2 = max(min(z2, INT*z3), (1-INT)*z3)
# #             z1 = z1 + z2;                                             % update the step
                z1 = z1 + z2
# #             X = X + z2*s;
                X = (X + z2*s)
# #             [f2 df2] = eval(argstr);
                f2, df2 = f(X)
# #             M = M - 1; i = i + (length<0);                             % count epochs?!
                M -= 1
                i += (length < 0)
# #             d2 = df2'*s;
                d2 = df2.dot(s)
# #             z3 = z3-z2;                      % z3 is now relative to the location of z2
                z3 = z3-z2
# #         end

# #         if f2 > f1+z1*RHO*d1 | d2 > -SIG*d1
            if (f2 > f1+z1*RHO*d1) or (d2 > -SIG*d1):
# #             break;                                                  % this is a failure
                break
# #         elseif d2 > SIG*d1
            elif d2 > SIG*d1:
# #             success = 1; break;                                               % success
                success = 1
                break
# #         elseif M == 0
            elif M == 0:
# #             break;                                                            % failure
                break
# #         end

# #         A = 6*(f2-f3)/z3+3*(d2+d3);                          % make cubic extrapolation
            A = 6*(f2-f3)/z3+3*(d2+d3)
# #         B = 3*(f3-f2)-z3*(d3+2*d2);
            B = 3*(f3-f2)-z3*(d3+2*d2)
# #         z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));            % num. error possible - ok!
            z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3))
# #         if ~isreal(z2) | isnan(z2) | isinf(z2) | z2 < 0       % num prob or wrong sign?
            if (not np.isreal(z2)) or np.isnan(z2) or np.isinf(z2) or (z2 < 0):
# #             if limit < -0.5                                 % if we have no upper limit
                if limit < -0.5:
# #                 z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
                    z2 = z1 * (EXT-1)
# #             else
                else:
# #                 z2 = (limit-z1)/2;                                   % otherwise bisect
                    z2 = (limit-z1)/2
# #             end

# #         elseif (limit > -0.5) & (z2+z1 > limit)              % extraplation beyond max?
            elif (limit > -0.5) and (z2+z1 > limit):
# #             z2 = (limit-z1)/2;                                                 % bisect
                z2 = (limit-z1)/2
# #         elseif (limit < -0.5) & (z2+z1 > z1*EXT)           % extrapolation beyond limit
            elif (limit < -0.5) and (z2+z1 > z1*EXT):
# #             z2 = z1*(EXT-1.0);                             % set to extrapolation limit
                z2 = z1*(EXT-1.0)
# #         elseif z2 < -z3*INT
            elif z2 < -z3*INT:
# #             z2 = -z3*INT;
                z2 = -z3*INT
# #         elseif (limit > -0.5) & (z2 < (limit-z1)*(1.0-INT))       % too close to limit?
            elif (limit > -0.5) & (z2 < (limit-z1)*(1.0-INT)):
# #             z2 = (limit-z1)*(1.0-INT);
                z2 = (limit-z1)*(1.0-INT)
# #         end

# #         f3 = f2; d3 = d2; z3 = -z2;                      % set point 3 equal to point 2
            f3 = f2
            d3 = d2
            z3 = -z2
# #         z1 = z1 + z2; X = X + z2*s;                          % update current estimates
            z1 += z2
            X = (X + z2*s)
# #         [f2 df2] = eval(argstr);
            f2, df2 = f(X)
# #         M = M - 1; i = i + (length<0);                                 % count epochs?!
            M -= 1
            i += (length < 0)
# #         d2 = df2'*s;
            d2 = df2.dot(s)
# #     end                                                            % end of line search
# #
# #     if success                                               % if line search succeeded
        if success:
# #         f1 = f2; fX = [fX' f1]';
            f1 = f2
            fX.append(f1)
# #         fprintf('%s %6i;  Value %4.6e\r', S, i, f1);
            print('{} {:>6}:  Value {:4.6e}'.format(S, i, f1))
# #         s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;          % Polack-Ribiere direction
            s = (df2.dot(df2)-df1.dot(df2))/(df1.dot(df1))*s - df2
# #         tmp = df1; df1 = df2; df2 = tmp;                             % swap derivatives
            tmp = df1
            df1 = df2
            df2 = tmp
# #         d2 = df1'*s;
            d2 = df1.dot(s)
# #         if d2 > 0                                          % new slope must be negative
            if d2 > 0:
# #             s = -df1;                                % otherwise use steepest direction
                s = -df1
# #             d2 = -s'*s;
                d2 = -s.dot(s)
# #         end

# #         z1 = z1 * min(RATIO, d1/(d2-realmin));              % slope ratio but max RATIO
            z1 = z1 * min(RATIO, d1/(d2-np.finfo(d2.dtype).eps))
# #         d1 = d2;
            d1 = d2
# #         ls_failed = 0;                                  % this line search did not fail
            ls_failed = 0
# #     else
        else:
# #         X = X0; f1 = f0; df1 = df0;      % restore point from before failed line search
            X = X0
            f1 = f0
            df1 = df0
# #         if ls_failed | i > abs(length)              % line search failed twice in a row
            if ls_failed or (i > abs(length)):
# #             break;                               % or we ran out of time, so we give up
                break
# #         end

# #         tmp = df1; df1 = df2; df2 = tmp;                             % swap derivatives
            tmp = df1
            df1 = df2
            df2 = tmp
# #         s = -df1;                                                        % try steepest
            s = -df1
# #         d1 = -s'*s;
            d1 = -s.dot(s)
# #         z1 = 1/(1-d1);
            z1 = 1/(1-d1)
# #         ls_failed = 1;                                        % this line search failed
            ls_failed = 1
# #     end
# # end
# # fprintf('\n');
    print("\n")

    return X, np.array(fX), i


def main():
    pass


if __name__ == "__main__":
    main()
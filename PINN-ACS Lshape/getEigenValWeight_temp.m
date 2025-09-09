function [eigenVals, weights , Losses_curve , eigenVal_curve] = getEigenValWeight_temp(eigen_guesses, x_vec, prevPop, ...
    A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2)
    
    tol = 1e-8;
    n = 500;
  

    numGuesses = length(eigen_guesses);
    eigenVals = zeros(1, numGuesses);
    weights = zeros(n, numGuesses);


    numPrev = size(prevPop, 2);  % Number of previous eigenvectors (n x p)
    x_ortho = zeros(numPrev, n);
    y_ortho = zeros(numPrev ,1);
   
    %% Adding orthogonality
    if numPrev > 0
        Phi = cos(w1 * (x_vec') + b1)';  % [m x n]
        for k = 1:numPrev
            prevW2 = prevPop(:, k);           % [n x 1] (previous weight vector)
            u_j = (Phi * prevW2 );           % [m x 1] (evaluated u_j at collocation)
            x_ortho(k,:) =(u_j') *Phi ;  % [n x n]
        end
    end
    %  storing the losses and eigen value convergence trend
    Losses_curve = zeros(MaxIter ,numGuesses);
    eigenVal_curve = zeros(MaxIter ,numGuesses);
    % Loop over each guess
    for g = 1:numGuesses
        eigenVal = eigen_guesses(g);
        w2final = inf;
        for iter = 1:MaxIter
            w2 = inf;
            minLoss = inf;
            res = A - eigenVal * B;
            % storing the current eigenvalue
            eigenVal_curve(iter ,g) = eigenVal;
            for j = 1:size(x_amp, 1) 
            
                Xc = [res;  sqrt(100)*x_bc;1e1*x_amp(j, :) ;1e2* x_ortho];
                Yc = [zeros(m2, 1); sqrt(100)*y_bc; 1e1* y_amp(j) ; 1e2*y_ortho];  
                X = ((Xc') * Xc) ;
                wopt = X\((Xc') * Yc);
                %  current loss
                loss = norm(Xc * wopt - Yc) ;
    
                if loss < minLoss
                    minLoss = loss;
                    w2 = wopt;
                    
                end
            end
            % storing the current Loss
            Losses_curve(iter,g) = minLoss;

            % Rayleigh update
            num = (A * w2)' * (B * w2) + (B * w2)' * (A * w2);
            den = 2 * (B * w2)' * (B * w2);
            newApproxEigen = num / den;
          
            eigenVal = newApproxEigen;
            w2final = w2;
        end
        % Store result for this guess
        eigenVals(g) = eigenVal;
        weights(:, g) = w2final;
    end
end

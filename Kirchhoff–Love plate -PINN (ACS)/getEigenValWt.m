function [eigenVals, weights , Losses_curve , eigenVal_curve] = getEigenValWt(eigen_guesses, x_vec, prevPop, ...
    A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2,n)
    

    numGuesses = length(eigen_guesses);
    eigenVals = zeros(1, numGuesses);
    weights = zeros(n, numGuesses);
       %  storing the losses and eigen value convergence trend
    Losses_curve = zeros(MaxIter ,numGuesses);
    eigenVal_curve = zeros(MaxIter ,numGuesses);
%% orthogonality incorporation
    numPrev = size(prevPop,2);
    x_ortho = zeros(numPrev, n);
    y_ortho = zeros(numPrev, 1);
    if numPrev > 0
        Phi = cos(w1*(x_vec') + b1)';
        for k = 1:numPrev
            prevW2 = prevPop(:, k);
            u_j = Phi * prevW2;
            x_ortho(k,:) =  (u_j') *Phi ;
        end
    end
    % Loop over each guess
    for g = 1:numGuesses
        eigenVal = eigen_guesses(g);
        w2final = inf;
        for iter = 1:MaxIter
            w2 = inf;
            minLoss = inf;
            res = A - eigenVal * B;
            eigenVal_curve(iter ,g) = eigenVal;
            for j = 1:size(x_amp, 1)

                Xc = [res; 1e1 * x_bc;  1e1*x_amp(j, :) ; 1e1*x_ortho];
                Yc = [zeros(m2, 1); 1e1 * y_bc;  1e1*y_amp(j); 1e1*y_ortho];

                X = (Xc' * Xc) ;
                wopt = X \ (Xc' * Yc);
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

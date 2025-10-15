
function plot_(final_lambdas, final_weights, w1, b1, loss_convergence, eigen_convergence, analyticalEigenVals, a, b, gridSize)
    numModes = min(10, size(final_weights, 2));

    % Create grid
    x = linspace(0, a, gridSize);
    y = linspace(0, b, gridSize);
    [X, Y] = meshgrid(x, y);
    xy_vec = [X(:)'; Y(:)'];  % 2 x (gridSize^2)

    % Compute global colormap limits
    allData = [];
    for i = 1:numModes
        w2 = final_weights(:, i);
        phi_valid = cos(w1 * xy_vec + b1);
        z_valid = (w2' * phi_valid)';
        z_valid = z_valid / norm(z_valid);
        allData = [allData; z_valid];
    end
    global_clim = [min(allData), max(allData)];

    % --- Plot modes ---
    for i = 1:numModes
        w2 = final_weights(:, i);
        phi_valid = cos(w1 * xy_vec + b1);
        z_valid = (w2' * phi_valid)';
        z_valid = z_valid / norm(z_valid);

        % Reshape for grid
        Z_nn = reshape(z_valid, gridSize, gridSize);

        figure;
        surf(X, Y, Z_nn, 'EdgeColor', 'none');
        view(2); axis equal tight; shading interp;
        colorbar; colormap jet; caxis(global_clim);
        xlabel('x'); ylabel('y');

        % Show predicted eigenvalue
        title(sprintf('Mode %d, Predicted Eigenvalue: %.4f', i, final_lambdas(i)));

        print(gcf, sprintf('NN_Mode_%d.svg', i), '-dsvg'); 
        close;
    end

   % --- Save loss convergence (no display) ---
   for i = 1:numModes
        f = figure('Visible','off');  % invisible figure
        semilogy(loss_convergence{i}, 'LineWidth', 1.5);
        grid on;
        xlabel('Iteration'); ylabel('Loss (log scale)');
        title(sprintf('Loss Convergence - Mode %d', i));
        print(f, sprintf('Loss_Convergence_Mode_%d.svg', i), '-dsvg');
        close(f);
    end

     % --- Eigenvalue convergence plots ---
     for i = 1:numModes
        figure('Color','w'); hold on;

        % Predicted eigenvalues
        predVals = eigen_convergence{i};
        plot(predVals, 'LineWidth', 3, 'Color', [0 0 0.55], ...
             'DisplayName', 'Convergence Trend');

        % True eigenvalue line (if available)
        if length(analyticalEigenVals) >= i
            trueVal = analyticalEigenVals(i);
            yline(trueVal, '--', 'Color', [0.85 0 0], ...
                  'LineWidth', 3, 'DisplayName', 'True Eigenvalue');
        end

        % Style: dark box, grid, thicker axes
        grid on; box on;
        set(gca, 'FontSize', 14, 'LineWidth', 2, ...   % thicker axis lines
                 'XColor', [0 0 0], 'YColor', [0 0 0]);

        xlabel('Iteration');
        ylabel('Eigenvalue Estimate');
        title(sprintf('Eigenvalue Convergence - Mode %d', i));

        % Show legend only on first plot
        if i == 1
            legend('show', 'FontSize', 23, 'Location', 'best');
        end
    end
    

end

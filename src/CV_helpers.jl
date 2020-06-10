using Base.Threads

function make_grid(α, β)
    return [ a for a in collect(Iterators.product(α, β)) if a[1] + a[2] < 1]
end

function CV_LS(y, splits, DA, grid, num_CV_trials, ε, max_iterations=max_iterations)
    num_grid_pts = length(grid)
    CV_accs = zeros(Float64, num_grid_pts)
    n = length(y)
    @sync for i = 1:num_grid_pts
        Threads.@spawn begin
            α = grid[i]
            # Get accuracy over each split for this hyperparameter setting
            hyperparam_accs = Float64[]
            for split in splits
                # Run label spreading for each class
                X_LS_labels = zeros(n, length(split))
                for (class, class_split) in enumerate(split)
                    train_inds = class_split[1]
                    Y = zeros(Float64, n)
                    Y[train_inds] .= 1.0
                    Y = (1 - ε) .* Y .+ ε
                    X_LS_labels[:, class], _ = Tensor_Package.standard_label_spreading(x->DA*x,Y,α,1-α, max_iterations=max_iterations)
                end
                # Evaluate accuracy on this split
                test_inds = []
                for (class, class_split) in enumerate(split)
                    append!(test_inds, class_split[2])
                end
                Y_pred = map(x->x[2], argmax(X_LS_labels[test_inds,:], dims=2))
                Y_real = y[test_inds]
                acc = Tensor_Package.accuracy(Y_pred, Y_real)
                push!(hyperparam_accs, acc)
            end
            # Just record average accuracy
            CV_accs[i] = mean(hyperparam_accs)
        end
    end

    # Now return best parameters (best mean accuracy)
    return grid[argmax(CV_accs)]
end

function validation_metrics(y, training_inds, DA, α, ε, max_iterations)
    n = length(y)
    X_LS_labels = zeros(n, length(training_inds))
    for (class, class_inds) in enumerate(training_inds)
        Y = zeros(Float64, n)
        Y[class_inds] .= 1.0
        Y = (1 - ε) .* Y .+ ε
        X_LS_labels[:, class], _ = Tensor_Package.standard_label_spreading(x->DA*x,Y,α,1-α, max_iterations=max_iterations)
    end

    test_inds = ones(Bool, n)
    for class_inds in training_inds
        test_inds[class_inds] .= false
    end
    test_inds = findall(test_inds)

    Y_pred = map(x->x[2], argmax(X_LS_labels[test_inds,:], dims=2))
    Y_real = y[test_inds]
    return [Tensor_Package.accuracy(Y_pred, Y_real), Tensor_Package.recall(Y_pred, Y_real), Tensor_Package.precision(Y_pred, Y_real)]
end

function CV_HOLS(y, splits, DA, DT, B, grid, num_CV_trials, f, ε, max_iterations)
    num_grid_pts = length(grid)
    CV_accs = zeros(Float64, num_grid_pts)
    n = length(y)
    @sync for i = 1:num_grid_pts
        Threads.@spawn begin
            (α, β) = grid[i]
            # Get accuracy over each split for this hyperparameter setting
            hyperparam_accs = zeros(Float64, length(splits))
            @sync for (ind, split) in enumerate(splits)
                Threads.@spawn begin
                    # Run label spreading for each class
                    X_HOLS_labels = zeros(n, length(split))
                    for (class, class_split) in enumerate(split)
                        train_inds = class_split[1]
                        Y = zeros(Float64, n)
                        Y[train_inds] .= 1.0
                        Y = (1 - ε) .* Y .+ ε
                        tildeY = copy(Y)
                        if Tensor_Package.φ(tildeY,f,B) > 1e-20
                            tildeY = tildeY ./ Tensor_Package.φ(tildeY,f,B)
                        end
                        X_HOLS_labels[:, class], _ = Tensor_Package.projected_second_order_label_spreading(
                            x->Tensor_Package.Tf(DT,f,x),
                            x->DA*x,
                            tildeY,
                            α,β,1-α-β,x->Tensor_Package.φ(x,f,B), max_iterations=max_iterations)
                    end
                    # Evaluate accuracy on this split
                    test_inds = []
                    for (class, class_split) in enumerate(split)
                        append!(test_inds, class_split[2])
                    end
                    Y_pred = map(x->x[2], argmax(X_HOLS_labels[test_inds,:], dims=2))
                    Y_real = y[test_inds]
                    acc = Tensor_Package.accuracy(Y_pred, Y_real)
                    hyperparam_accs[ind] = acc
                end
            end
            # Just record average accuracy
            CV_accs[i] = mean(hyperparam_accs)
        end
    end

    # Now return best parameters (best mean accuracy)
    return grid[argmax(CV_accs)]
end

function validation_metrics(y, training_inds, DA, DT, B, α, β, f, ε, max_iterations)
    n = length(y)
    X_HOLS_labels = zeros(n, length(training_inds))
    for (class, class_inds) in enumerate(training_inds)
        Y = zeros(Float64, n)
        Y[class_inds] .= 1.0
        Y = (1 - ε) .* Y .+ ε
        tildeY = copy(Y)
        if Tensor_Package.φ(tildeY,f,B) > 1e-20
            tildeY = tildeY ./ Tensor_Package.φ(tildeY,f,B)
        end
        X_HOLS_labels[:, class], _ = Tensor_Package.projected_second_order_label_spreading(
            x->Tensor_Package.Tf(DT,f,x),
            x->DA*x,
            tildeY,
            α,β,1-α-β,x->Tensor_Package.φ(x,f,B), max_iterations=max_iterations)
    end

    test_inds = ones(Bool, n)
    for class_inds in training_inds
        test_inds[class_inds] .= false
    end
    test_inds = findall(test_inds)

    Y_pred = map(x->x[2], argmax(X_HOLS_labels[test_inds,:], dims=2))
    Y_real = y[test_inds]
    return [Tensor_Package.accuracy(Y_pred, Y_real), Tensor_Package.recall(Y_pred, Y_real), Tensor_Package.precision(Y_pred, Y_real)]
end

# default 50/50 split
function CV_splits(training_inds, num_splits, split=0.5)
    all_splits = []
    for _ in 1:num_splits
        trial_splits = []
        for inds in training_inds
            split_ind = Int64.(ceil(length(inds) * split))
            class_splits = []
            shuffle!(inds)
            push!(trial_splits, (inds[1:split_ind], inds[(split_ind + 1):end]))
        end
        push!(all_splits, trial_splits)
    end
    return all_splits
end


function CV_binary(DA, DT, B, y, mixing_functions, method_names, num_CV_trials,
                   percentage_of_known_labels;
                   balanced=true, noise = 0, ε=1e-6,
                   ho_search_params = make_grid(0.8, 0.8),
                   ls_search_params = 0.1:0.1:.9,
                   max_iterations = 100)
    num_per_class = Tensor_Package.generate_known_labels(percentage_of_known_labels, balanced, y)
    training_inds = []
    for (label, num) in enumerate(num_per_class)
        class_inds = findall(y .== label)
        shuffle!(class_inds)
        push!(training_inds, class_inds[1:num])
    end
    splits = CV_splits(training_inds, num_CV_trials)

    df = DataFrame

    αs = Float64[]
    βs = Float64[]
    accs = Float64[]
    recs = Float64[]
    precs = Float64[]
    names = []

    @time begin
        for (j, (f, method_name)) in enumerate(zip(mixing_functions, method_names))
            println("$method_name...")
            if j < length(mixing_functions)
                # CV to get best α and β parameters
                α_best, β_best = CV_HOLS(y, splits, DA, DT, B, ho_search_params, num_CV_trials, f, ε, max_iterations)
                # Now evaluate on the entire data
                acc, rec, prec = validation_metrics(y, training_inds, DA, DT, B, α_best, β_best, f, ε, max_iterations)
                push!(αs, α_best)
                push!(βs, β_best)
                push!(accs, acc)
                push!(precs, prec)
                push!(recs, rec)
            else
                # CV to get best α parameter
                α_best = CV_LS(y, splits, DA, ls_search_params, num_CV_trials, ε, max_iterations)
                # Now evaluate on the entire data
                acc, rec, prec = validation_metrics(y, training_inds, DA, α_best, ε, max_iterations)
                push!(αs, α_best)
                push!(βs, 0.0)
                push!(accs, acc)
                push!(precs, prec)
                push!(recs, rec)
            end
            push!(names, method_name)
        end
    end

    final_data = DataFrame()
    final_data[!, :α] = αs
    final_data[!, :β] = βs
    final_data[!, :acc] = accs
    final_data[!, :rec] = recs
    final_data[!, :prec] = precs
    final_data[!, :method_name] = names
    final_data[!, :size] .= length(y)
    final_data[!, :percentage_of_known_labels] .= percentage_of_known_labels
    final_data[:, :balanced] .= balanced
    final_data[:, :noise] .= noise
    #return final_data
    return (final_data, training_inds)
end

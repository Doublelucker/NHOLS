using UCIData, Random, LinearAlgebra, SparseArrays, Distances, DataFrames, CSV, LightGraphs, MAT, Statistics

using Base.Threads

include("structs.jl")

#export generate_known_labels, φ, projected_second_order_label_spreading, standard_label_spreading

function projected_second_order_label_spreading(Tfun,Afun,y,α,β,γ,φ; starting_vector = "all_ones", max_iterations = 200, tolerance = 1e-5, verbose = false)
    ## Input: Tfun = function that performs the product Tf(x)
    ##        Afun = funciton that perfroms the product A*x
    #     c = size(Y,2); #number of classes

    if starting_vector == "all_ones"
        x_0 = ones(length(y))[:];
        # x_0 = x_0 ./ φ(x_0)
    else
        x_0 = starting_vector;
    end

    error_sequence = [];
    x_new = [];
    φ_values = []
    for k in 1 : max_iterations
        xx_new =  α .* Tfun(x_0) + β .* Afun(x_0) + γ .* y;
        x_new = xx_new ./ φ(xx_new);
        append!(φ_values, φ(xx_new))
        error_sequence = [error_sequence; norm(x_new - x_0) / norm(x_new)];
        if error_sequence[end] < tolerance
            return x_new, error_sequence, k, φ_values
        end

        if φ(xx_new)==NaN
            error("xx_new  or φ(xx_new) are NaN")
        end

        x_0 = copy(x_new);
        if verbose
            @show k
        end
    end
    if verbose
        println("Reached max number of iterations without convergence")
    end
    return x_new, error_sequence, max_iterations, φ_values
end

function standard_label_spreading(Afun,y,α,β; starting_vector = "all_ones", max_iterations = 200, tolerance = 1e-5, verbose = false)
    ## Input: Afun = funciton that perfroms the product A*x

    if starting_vector == "all_ones"
        x_0 = ones(length(y))[:];
    else
        x_0 = starting_vector;
    end

    error_sequence = [];
    x_new = [];
    for k in 1 : max_iterations
        x_new =  α .* Afun(x_0) + β .* y;
        #x_new = xx_new ./ φ(xx_new, B);
        error_sequence = [error_sequence; norm(x_new - x_0) / norm(x_new)];
        if error_sequence[end] < tolerance
            return x_new, error_sequence, k
        end
        x_0 = copy(x_new);
        if verbose @show k end
    end
    if verbose println("Reached max number of iterations without convergence") end
    return x_new, error_sequence, max_iterations
end


function generate_known_labels(percentage_of_known_labels, balanced, ground_truth_classes)
    number_of_classes = length(unique(ground_truth_classes))
    num_per_class = [sum(ground_truth_classes .== i) for i in 1:number_of_classes]

    if balanced
        known_labels_per_each_class = Int64.(ceil.(percentage_of_known_labels.*num_per_class'))
    else
        known_labels_per_each_class = percentage_of_known_labels.*num_per_class'
        known_labels_per_each_class = Int64.(ceil.(known_labels_per_each_class .+ (known_labels_per_each_class./2).*randn(size(known_labels_per_each_class)) ))
        known_labels_per_each_class = min.(known_labels_per_each_class, num_per_class')
        known_labels_per_each_class = max.(known_labels_per_each_class,1)
    end

    return known_labels_per_each_class
end

function analyze_prediction(A, DG_isqrt, T, DH_isqrt,B,φ,
                                                       mixing_functions,
                                                       percentage_of_known_labels,
                                                       true_classes; balanced = true, ε = 1e-6, α = .4, β = .3)
    num_of_classes = length(unique(true_classes))
    known_labels_per_each_class = generate_known_labels(percentage_of_known_labels, balanced, true_classes)
    n = size(A,1)
    num_of_methods = length(mixing_functions)+1

    pred_accuracy = zeros(length(percentage_of_known_labels), num_of_methods) # SortedDict()
    pred_precision = zeros(length(percentage_of_known_labels), num_of_methods) # SortedDict()
    pred_recall = zeros(length(percentage_of_known_labels), num_of_methods) # SortedDict()

    X_labels = zeros(num_of_methods, n, num_of_classes) #   SortedDict()

    @threads for i in 1:length(percentage_of_known_labels)
        #println("$p...")

        row_index_class = 0

        @threads for class in 1:num_of_classes
            #println("$class...")

            num_known = known_labels_per_each_class[i,class]
            in_class = findall(vec(true_classes) .== class)
            shuffle!(in_class)
            pos_labels = in_class[1:num_known]

            Y = zeros(Float64, n)
            Y[pos_labels] .= 1.0
            Y = (1 - ε) .* Y .+ ε

            for (j,f) in enumerate(mixing_functions)
                tildeY = Y
                if φ(DH_isqrt .* tildeY,f,B) > 1e-20
                    tildeY = tildeY ./ φ(tildeY,f,B)
                end # it seems to work slightly better if we start directly with a normalized Y, when possible
                X_labels[j,:,class], _ = projected_second_order_label_spreading(
                    x -> Tf(T, DH_isqrt, f, x),
                    x -> Ax(A, DG_isqrt, x),
                    tildeY,
                    α,β,1-α-β,x->φ(DH_isqrt .* x,f,B));

            end

            X_labels[end,:,class], _ = standard_label_spreading(x->DA*x,Y,α+β,1-α-β);
        end

        for j in 1:num_of_methods
            Y_predicted = map(x->x[2], argmax(X_labels[j,:,:],dims=2))
            pred_accuracy[i,j] = accuracy(Y_predicted, true_classes)
            pred_precision[i,j] = precision(Y_predicted, true_classes)
            pred_recall[i,j] = recall(Y_predicted, true_classes)

        end

        # for (X,method) in zip(X_labels,method_names)
        #     Y_predicted = map(x->x[2], argmax(X[2],dims=2))
        #     push!(pred_accuracy[p][method], accuracy(Y_predicted, true_classes))
        # end
    end

    return pred_accuracy,pred_precision,pred_recall
end

function analyze_prediction_LS(A, DG_isqrt, percentage_of_known_labels, true_classes; balanced = true, ε = 1e-6, α = .4)
    num_of_classes = length(unique(true_classes))
    known_labels_per_each_class = generate_known_labels(percentage_of_known_labels, balanced, true_classes)
    n = size(A,1)

    pred_accuracy = zeros(length(percentage_of_known_labels))
    pred_precision = zeros(length(percentage_of_known_labels))
    pred_recall = zeros(length(percentage_of_known_labels))

    X_labels = zeros(n, num_of_classes)
    @threads for i in 1:length(percentage_of_known_labels)
        #println("$p...")

        row_index_class = 0

        @threads for class in 1:num_of_classes
            #println("$class...")

            num_known = known_labels_per_each_class[i,class]
            in_class = findall(vec(true_classes) .== class)
            shuffle!(in_class)
            pos_labels = in_class[1:num_known]

            Y = zeros(Float64, n)
            Y[pos_labels] .= 1.0
            Y = (1 - ε) .* Y .+ ε
            X_labels[:,class], _ = standard_label_spreading(x -> Ax(A, DG_isqrt, x),Y,α,1-α);
        end

        Y_predicted = map(x->x[2], argmax(X_labels,dims=2))
        pred_accuracy[i] = accuracy(Y_predicted, true_classes)
        pred_precision[i] = precision(Y_predicted, true_classes)
        pred_recall[i] = recall(Y_predicted, true_classes)

        # for (X,method) in zip(X_labels,method_names)
        #     Y_predicted = map(x->x[2], argmax(X[2],dims=2))
        #     push!(pred_accuracy[p][method], accuracy(Y_predicted, true_classes))
        # end
    end

    return pred_accuracy,pred_precision,pred_recall
end

function analyze_prediction_HOLS(A, DG_isqrt, T, DH_isqrt,B,φ, mixing_functions, percentage_of_known_labels, true_classes; balanced = true, ε = 1e-6, α = .4, β = .3)
    num_of_classes = length(unique(true_classes))
    known_labels_per_each_class = generate_known_labels(percentage_of_known_labels, balanced, true_classes)
    n = size(A,1)
    num_of_methods = length(mixing_functions)

    pred_accuracy = zeros(length(percentage_of_known_labels), num_of_methods) # SortedDict()
    pred_precision = zeros(length(percentage_of_known_labels), num_of_methods) # SortedDict()
    pred_recall = zeros(length(percentage_of_known_labels), num_of_methods) # SortedDict()

    X_labels = zeros(num_of_methods, n, num_of_classes) #   SortedDict()

    @threads for i in 1:length(percentage_of_known_labels)
        #println("$p...")

        row_index_class = 0

        @threads for class in 1:num_of_classes
            #println("$class...")

            num_known = known_labels_per_each_class[i,class]
            in_class = findall(vec(true_classes) .== class)
            shuffle!(in_class)
            pos_labels = in_class[1:num_known]

            Y = zeros(Float64, n)
            Y[pos_labels] .= 1.0
            Y = (1 - ε) .* Y .+ ε

            for (j,f) in enumerate(mixing_functions)
                tildeY = Y
                if φ(DH_isqrt .* tildeY,f,B) > 1e-20
                    tildeY = tildeY ./ φ(tildeY,f,B)
                end # it seems to work slightly better if we start directly with a normalized Y, when possible
                X_labels[j,:,class], _ = projected_second_order_label_spreading(
                    x -> Tf(T, DH_isqrt, f, x),
                    x -> Ax(A, DG_isqrt, x),
                    tildeY,
                    α,β,1-α-β,x->φ(DH_isqrt .* x,f,B));

            end
        end

        for j in 1:num_of_methods
            Y_predicted = map(x->x[2], argmax(X_labels[j,:,:],dims=2))
            pred_accuracy[i,j] = accuracy(Y_predicted, true_classes)
            pred_precision[i,j] = precision(Y_predicted, true_classes)
            pred_recall[i,j] = recall(Y_predicted, true_classes)

        end
    end

    return pred_accuracy, pred_precision, pred_recall
end


function analyze_dataset(dataset_name,num_trials,
                                    knn,
                                    A, DG_isqrt, T, DH_isqrt,B,φ,
                                    mixing_functions,
                                    percentage_of_known_labels,
                                    true_classes,
                                    balanced,
                                    alphas,
                                    betas,
                                    ε)
    results = []
    method_names = 1:length(mixing_functions)+1
    for (α,β) in zip(alphas,betas)

        # pred_accuracy = zeros(length(percentage_of_known_labels), length(method_names))
        average_accuracy = zeros(length(percentage_of_known_labels), length(method_names))
        average_precision = zeros(length(percentage_of_known_labels), length(method_names))
        average_recall = zeros(length(percentage_of_known_labels), length(method_names))

        @threads for i in 1:num_trials
            println("\t trial no "*string(i))

            pred_accuracy,pred_precision,pred_recall = analyze_prediction(
                                    A, DG_isqrt, T, DH_isqrt,B,φ,
                                    mixing_functions,
                                    percentage_of_known_labels,
                                    true_classes,
                                    balanced = balanced,
                                    ε = ε, α = α, β = β)

            average_accuracy = average_accuracy .+ pred_accuracy
            average_precision = average_precision .+ pred_precision
            average_recall = average_recall .+ pred_recall
        end

        average_accuracy = average_accuracy ./ num_trials
        average_precision = average_precision ./ num_trials
        average_recall = average_recall ./ num_trials

        for i in 1 : size(average_accuracy,1)
            for j in 1 : length(mixing_functions)+1
                new_results =  Tuple([dataset_name;
                                      size(A,1);
                                      knn;
                                      α+β;
                                      α;
                                      β;
                                      percentage_of_known_labels[i]*100;
                                      balanced;
                                      average_accuracy[i,j];
                                      average_precision[i, j];
                                      average_recall[i,j] ] )
                push!(results,  new_results)
            end
        end

    end
    return results
end

function analyze_dataset_LS(dataset_name,num_trials,
                                    alphas,
                                    knn,
                                    A, DG_isqrt,
                                    percentage_of_known_labels,
                                    true_classes,
                                    balanced,
                                    ε)
    results = []
    for α in alphas

        # pred_accuracy = zeros(length(percentage_of_known_labels), length(method_names))
        average_accuracy = zeros(length(percentage_of_known_labels))
        average_precision = zeros(length(percentage_of_known_labels))
        average_recall = zeros(length(percentage_of_known_labels))
        @show num_trials
        @threads for i in 1:num_trials
            println("\t trial no "*string(i))

            pred_accuracy,pred_precision,pred_recall = analyze_prediction_LS(
                                    A, DG_isqrt,
                                    percentage_of_known_labels,
                                    true_classes,
                                    balanced = balanced,
                                    ε = ε, α = α)

            average_accuracy = average_accuracy .+ pred_accuracy
            average_precision = average_precision .+ pred_precision
            average_recall = average_recall .+ pred_recall
        end

        average_accuracy = average_accuracy ./ num_trials
        average_precision = average_precision ./ num_trials
        average_recall = average_recall ./ num_trials

        for i in 1 : size(average_accuracy,1)
            new_results =  Tuple([dataset_name;
                                  size(A,1);
                                  knn;
                                  1;
                                  α;
                                  1 - α;
                                  percentage_of_known_labels[i]*100;
                                  balanced;
                                  average_accuracy[i];
                                  average_precision[i];
                                  average_recall[i] ] )
            push!(results,  new_results)
        end

    end
    return results
end

function analyze_dataset_HOLS(dataset_name,num_trials,
                                    knn,
                                    A, DG_isqrt, T, DH_isqrt,B,φ,
                                    mixing_functions,
                                    percentage_of_known_labels,
                                    true_classes,
                                    balanced,
                                    alphas,
                                    betas,
                                    ε)
    results = []
    method_names = zeros(length(mixing_functions))
    for (α,β) in zip(alphas,betas)

        # pred_accuracy = zeros(length(percentage_of_known_labels), length(method_names))
        average_accuracy = zeros(length(percentage_of_known_labels), length(mixing_functions))
        average_precision = zeros(length(percentage_of_known_labels), length(mixing_functions))
        average_recall = zeros(length(percentage_of_known_labels), length(mixing_functions))
        for i in 1:num_trials
            println("\t trial no "*string(i))

            pred_accuracy,pred_precision,pred_recall = analyze_prediction_HOLS(
                                    A, DG_isqrt, T, DH_isqrt, B, φ,
                                    mixing_functions,
                                    percentage_of_known_labels,
                                    true_classes,
                                    balanced = balanced,
                                    ε = ε, α = α, β = β)

            average_accuracy = average_accuracy .+ pred_accuracy
            average_precision = average_precision .+ pred_precision
            average_recall = average_recall .+ pred_recall
        end

        average_accuracy = average_accuracy ./ num_trials
        average_precision = average_precision ./ num_trials
        average_recall = average_recall ./ num_trials

        for i in 1 : size(average_accuracy,1)
            for j in 1 : length(mixing_functions)
                new_results =  Tuple([j;
                                      dataset_name;
                                      size(A,1);
                                      knn;
                                      α+β;
                                      α;
                                      β;
                                      percentage_of_known_labels[i]*100;
                                      balanced;
                                      average_accuracy[i,j];
                                      average_precision[i,j];
                                      average_recall[i,j] ] )
                push!(results,  new_results)
            end
        end

    end
    return results
end


function φ(x, f, B::SparseArrays.SparseMatrixCSC)
    sum = 0.0
    for (i, j, v) in zip(findnz(B)...)
        @inbounds sum += v * (f(x[i], x[j]))^2
    end
    return 0.5 * sqrt(sum)
end

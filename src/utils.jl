
using UCIData, Random, LinearAlgebra, SparseArrays, Distances, DataFrames, CSV, LightGraphs, MAT, Statistics, PyCall

@pyimport numpy as np

export prepare_uci_data, accuracy, precision, recall, train_test_val_split, train_test_split


function accuracy(y_predicted, y_actual)
    return ( sum(y_predicted .== y_actual) ./ length(y_actual) )*100
end

function precision(y_predicted, y_actual; method="mean")
    if method=="mean"
        p = 0
        for label in unique(y_predicted)
            p += sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_predicted .== label)
        end
        p = p/length(unique(y_predicted)) *100
    elseif method=="min"
        p = Inf
        for label in unique(y_predicted)
            p = min(p, sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_predicted .== label) ) *100
        end
    else
        @assert false
    end
    return p
end

function recall(y_predicted, y_actual; method="mean")
    if method=="mean"
        p = 0
        for label in unique(y_predicted)
            p += sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_actual .== label)
        end
        p = p/length(unique(y_predicted)) *100
    elseif method=="min"
        p = Inf
        for label in unique(y_predicted)
            p = min(p, sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_actual .== label) ) *100
        end
    else
        @assert false
    end
    return p
end

function prepare_uci_data(dataset_name)
   dataset = UCIData.dataset(dataset_name)
   titles = unique(dataset.target)
   d = Dict(title => i  for (i, title) in zip(1:length(titles), titles))
   n_dataset = DataFrame(replace!(convert(Matrix, dataset), d...))
   rename!(n_dataset, names(dataset))
   y = n_dataset.target
   X = convert(Matrix, dataset[!, 2:end-1])
   X = convert(Array{Float64,2}, X)
   return X, y
end



function train_test_val_split(y, perc_train, perc_val; balanced=false)
        n = length(y)
        num_classes = length(unique(y))
        Y_test = zeros(n, num_classes)
        Y_train = zeros(n, num_classes)
        Y_val = zeros(n, num_classes)
        test_mask = zeros(n)
        train_mask = zeros(n)
        val_mask = zeros(n)
        num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
        num_val_per_class = Tensor_Package.generate_known_labels(perc_val, balanced, y)
        print(num_val_per_class)
        for (label, num_train, num_val) in zip(1:num_classes, num_train_per_class, num_val_per_class)
           class_inds = findall(y .== label)
           shuffle!(class_inds)

           train_indices = class_inds[1:num_train]
           Y_train[train_indices, label] .= 1
           train_mask[train_indices] .= 1

           test_indices = class_inds[num_train+num_val+1:end]
           Y_test[test_indices, label] .= 1
           test_mask[test_indices] .= 1

           val_indices = class_inds[num_train+1:num_train+num_val]
           Y_val[val_indices, label] .= 1
           val_mask[val_indices] .= 1

        end
        return np.array(Y_train), np.array(Y_test), np.array(Y_val), Bool.(train_mask), Bool.(test_mask), Bool.(val_mask)
end


function train_test_split(X, y, perc_train; balanced=false)
#minus(indx, x) = setdiff(1:length(y), indx)
        n = length(y)
        num_classes = length(unique(y))
        num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
        num_train_total = sum(num_train_per_class)
        Y_test = zeros(n - num_train_total, num_classes)
        Y_train = zeros(num_train_total, num_classes)
        full_train_indices = []
        full_test_indices = []
        for (label, num_train) in zip(1:num_classes, num_train_per_class)
           class_inds = findall(y .== label)
           shuffle!(class_inds)

           train_indices = class_inds[1:num_train]
           push!(full_train_indices, train_indices...)
           Y_train[(label-1)*num_train + 1:label*num_train, label] .= 1

           num_test = length(class_inds) - num_train
           test_indices = class_inds[num_train+1:end]
           push!(full_test_indices, test_indices...)
           Y_test[(label-1)*num_test + 1:label*num_test, label] .= 1

        end
        return np.array(Y_train), np.array(Y_test), X[full_train_indices, :], X[full_test_indices, :], X[vcat(full_train_indices, full_test_indices), :]
end

function parse_args(args)
   if (length(args) == 3)
      if (typeof(args[3]) == Int64)
         return range(args[1], args[2], length=args[3])
      else
         return args
      end
   else
      return args
   end
end

function prepare_config_data(data)
   balanced=data["balanced"]
   binary=data["binary"]
   alphas = parse_args(data["α"])
   ε = data["ε"]
   kn = data["kn"]
   noise = data["noise"]
   percentage_of_known_labels = parse_args(data["percentage_of_known_labels"])
   num_trials = data["num_trials"]
   dataset_name = data["dataset"]
   data_type = data["data_type"]

   betas = nothing
   if "β" in keys(data)
      betas = parse_args(data["β"])
   end

   distance = nothing
   if "distance" in keys(data)
      distance = getfield(Main, Symbol(data["distance"]))
   end

   mixing_functions = nothing
   if "mixing_functions" in keys(data)
      @show data["mixing_functions"]
      mixing_functions = map(x -> getfield(Main, Symbol(x)), split(data["mixing_functions"], ", "))
   end

   return balanced, binary, alphas, betas, distance, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type
end

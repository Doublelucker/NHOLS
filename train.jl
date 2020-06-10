
include("src\\utils.jl")
include("src\\tensors.jl")
include("src\\CV_helpers.jl")
include("src\\functions.jl")
include("src\\labelspreading.jl")

using YAML

data = YAML.load(open("config.yml"))

mode = data["mode"]

if !isdir("./results")
    mkdir("./results")
end

colnames = [:dataset_name, :size, :knn, :α_plus_β, :α, :β, :known_labels, :balanced, :acc, :prec, :rec]

if mode == "LS"
    balanced, binary, alphas, _, _, _, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    weight_function = nothing
    features, y, A, DG_isqrt = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    data = analyze_dataset_LS(dataset_name,num_trials,
                                        alphas,
                                        kn,
                                        A, DG_isqrt,
                                        percentage_of_known_labels,
                                        y,
                                        balanced,
                                        ε)
    df = DataFrame(data)
    rename!(df, colnames)
    CSV.write("./results/results_$(dataset_name)_$(mode).csv", df)

elseif (mode == "HOLS")
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    data = analyze_dataset_HOLS(dataset_name,num_trials,
                                        kn,
                                        A, DG_isqrt, T, DH_isqrt,B,φ,
                                        mixing_functions,
                                        percentage_of_known_labels,
                                        y,
                                        balanced,
                                        alphas,
                                        betas,
                                        ε)

    df = DataFrame(data)
    rename!(df, colnames)
    CSV.write("./results/results_$(dataset_name)_$(mode).csv", df)

elseif (mode == "both")
    balanced, binary, alphas, betas, weight_function, mixing_functions, ε, kn, noise, percentage_of_known_labels, num_trials, dataset_name, data_type = prepare_config_data(data[mode])
    features, y, A, DG_isqrt, T, DH_isqrt, B = load_data(dataset_name, kn, noise, weight_function, mode, binary)
    data = analyze_dataset(dataset_name,num_trials,
                                            kn,
                                            A, DG_isqrt, T, DH_isqrt, B,φ,
                                            mixing_functions,
                                            percentage_of_known_labels,
                                            y,
                                            balanced,
                                            alphas,
                                            betas,
                                            ε)

    df = DataFrame(data)
    rename!(df, colnames)
    CSV.write("./results/results_$(dataset_name)_$(mode).csv", df)
end

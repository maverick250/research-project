import torch
import torch.nn as nn

class BiLSTM_Base(nn.Module):
    """
    A Bidirectional LSTM (BiLSTM) neural network module for embedding input data.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, GPU=False):
        super(BiLSTM_Base, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim / 2)  # Bidirectional LSTM splits hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        self.GPU = GPU

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)  # Initial hidden state
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)  # Initial cell state

        if self.GPU:
            h0 = h0.to("cuda:0")
            c0 = c0.to("cuda:0")

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        return out


class StickBreakingProcess(nn.Module):
    """
    Stick-breaking process to assign dynamic clusters using DPMM.
    """
    
    def __init__(self, alpha):
        super(StickBreakingProcess, self).__init__()
        self.alpha = alpha

    def forward(self, z):
        # Sample Beta distribution for stick-breaking process
        beta_dist = torch.distributions.Beta(1, self.alpha)
        betas = beta_dist.sample((z.size(0), z.size(-1))).to(z.device)
        
        # Stick-breaking formula
        remaining_stick = torch.cumprod(1 - betas, dim=-1)
        cluster_weights = betas * torch.cat([torch.ones_like(remaining_stick[:, :1]), remaining_stick[:, :-1]], dim=-1)
        
        return cluster_weights


class BiLSTM_Dynamic_Cluster(nn.Module):
    """
    LSTM embedding model with dynamic cluster assignment using stick-breaking process for DPMM.
    """

    def __init__(self, base_param_dict, lstm_param_dict, alpha=1.0, GPU=False):
        super(BiLSTM_Dynamic_Cluster, self).__init__()
        self.input_dim = base_param_dict["input_dim"]
        self.hidden_dim = lstm_param_dict["embed_hidden_dim"]
        self.num_lstm_layers_base = lstm_param_dict["num_lstm_layers_base"]
        self.latent_dim = base_param_dict["latent_dim"]
        self.alpha = alpha
        self.GPU = GPU

        # Base embedding LSTM
        self.base_model = BiLSTM_Base(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_lstm_layers_base,
            GPU=self.GPU)
        
        # Stick-breaking process for DPMM
        self.stick_breaking = StickBreakingProcess(alpha=self.alpha)

        # Mu and Sigma estimation models (latent variables for each cluster)
        self.mu_model = nn.Linear(self.hidden_dim * 2, self.latent_dim)
        self.sigma_model = nn.Linear(self.hidden_dim * 2, self.latent_dim)

    def forward(self, x):
        # Base LSTM embedding
        embedded_x = self.base_model(x)
        
        # Apply stick-breaking process for dynamic cluster assignment
        cluster_weights = self.stick_breaking(embedded_x)

        # Estimate latent variables (mu and sigma) for each cluster
        mu = self.mu_model(embedded_x)
        sigma = torch.exp(self.sigma_model(embedded_x))  # Use exp to ensure sigma is positive

        return {"cluster_weights": cluster_weights, "mu": mu, "sigma": sigma}

class LSTM_Embedder(nn.Module):
    """
    A neural network that uses LSTM for embedding input data and dynamically estimates
    cluster weights, mean (mu), and standard deviation (sigma) for each cluster using a
    stick-breaking process for DPMM.
    
    The input is a batch of sequences, and the output includes cluster weights, mu, and sigma
    for each sequence dynamically determined by DPMM.
    """

    def __init__(self, base_param_dict, lstm_param_dict, alpha=1.0, GPU=False):
        """
        Initializes the LSTM_Embedder with the specified parameters grouped in dictionaries.
        The number of clusters is dynamically determined using DPMM.

        Parameters:
            base_param_dict (dict): Dictionary containing base parameters such as input_dim and latent_dim.
            lstm_param_dict (dict): Dictionary containing LSTM-specific parameters such as embed_hidden_dim,
                                    num_lstm_layers_base, and num_lstm_layers_other.
            alpha (float): Concentration parameter for the Dirichlet Process.
            GPU (bool): Whether to utilize CUDA-capable GPUs for processing. Defaults to False.
        """
        super(LSTM_Embedder, self).__init__()
        self.input_dim = base_param_dict["input_dim"]
        self.hidden_dim = lstm_param_dict["embed_hidden_dim"]
        self.num_lstm_layers_base = lstm_param_dict["num_lstm_layers_base"]
        self.num_lstm_layers_other = lstm_param_dict["num_lstm_layers_other"]
        self.latent_dim = base_param_dict["latent_dim"]
        self.alpha = alpha
        self.GPU = GPU

        # Base LSTM for embedding
        self.base_model = BiLSTM_Base(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_lstm_layers_base,
            GPU=self.GPU
        )

        # Stick-breaking process for dynamic cluster assignment
        self.stick_breaking = StickBreakingProcess(alpha=self.alpha)

        # Mu and Sigma estimation models (latent variables for each dynamically assigned cluster)
        self.mu_model = nn.Linear(self.hidden_dim * 2, self.latent_dim)
        self.sigma_model = nn.Linear(self.hidden_dim * 2, self.latent_dim)

    def forward(self, x):
        """
        Defines the forward pass of the model. It embeds the input using the base LSTM,
        dynamically assigns clusters using the stick-breaking process, and then estimates mu and sigma
        for each cluster.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, time_sequence, input_dim).

        Returns:
            dict: A dictionary containing the outputs 'cluster_weights', 'mu', and 'sigma'.
                  - 'cluster_weights': Tensor representing the dynamic cluster assignments.
                  - 'mu': Tensor representing the estimated means for each cluster.
                  - 'sigma': Tensor representing the estimated standard deviations for each cluster.
        """
        # Base LSTM embedding
        embedded_x = self.base_model(x)

        # Stick-breaking process to dynamically assign clusters
        cluster_weights = self.stick_breaking(embedded_x)

        # Estimate mu and sigma for each dynamically assigned cluster
        mu = self.mu_model(embedded_x)
        sigma = torch.exp(self.sigma_model(embedded_x))  # Ensure sigma is positive using exp

        return {"cluster_weights": cluster_weights, "mu": mu, "sigma": sigma}

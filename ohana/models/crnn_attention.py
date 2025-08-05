import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
        Temporal Attention mechanism

    """
    def __init__(self, hidden_size):
        """
            Module learns to assign different weights to the LSTM's output at each
            time step, making it focus on the most relevant parts
            Attributes:
                hidden_size ()
        """
        super(Attention, self).__init__()
        self.attention_net = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        """
        Forward pass for the attention mechanism.
        
        Args:
            lstm_output (torch.Tensor): The output from the LSTM layer.
                                        Shape: (batch_size, seq_length, hidden_size)
                                        
        Returns:
            torch.Tensor: The context vector after applying attention.
                          Shape: (batch_size, hidden_size)
            torch.Tensor: The attention weights for each time step.
                          Shape: (batch_size, seq_length)
        """
        # (batch_size, seq_length, hidden_size)
        attn_energies = self.attention_net(lstm_output)
        attn_energies = torch.tanh(attn_energies)
        
        # (batch_size, seq_length, 1)
        attn_scores = self.context_vector(attn_energies)
        
        # (batch_size, seq_length)
        attn_weights = F.softmax(attn_scores.squeeze(2), dim=1)
        
        # (batch_size, 1, seq_length)
        attn_weights_unsqueezed = attn_weights.unsqueeze(1)
        
        # (batch_size, 1, hidden_size)
        context = torch.bmm(attn_weights_unsqueezed, lstm_output)
        
        # (batch_size, hidden_size)
        context = context.squeeze(1)
        
        return context, attn_weights

class CRNNAttention(nn.Module):
    """
    Convolutional Recurrent Neural Network with Attention.
    
    This model processes spatiotemporal data cubes (patches from the detector).
    1. A CNN encoder extracts spatial features from each frame independently.
    2. An LSTM processes the sequence of features to model temporal dependencies.
    3. An Attention mechanism weighs the importance of each time step.
    4. A final classifier predicts the event type.
    """
    def __init__(self, num_classes=4, cnn_output_size=256, lstm_hidden_size=128, num_lstm_layers=1, dropout=0.5):
        """
        Args:
            num_classes (int): Number of output classes (e.g., background, CR, snowball, RTN).
            cnn_output_size (int): The size of the feature vector from the CNN encoder.
            lstm_hidden_size (int): The number of features in the LSTM hidden state.
            num_lstm_layers (int): The number of recurrent layers in the LSTM.
            dropout (float): Dropout probability for regularization.
        """
        super(CRNNAttention, self).__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size

        # --- 1. Frame-wise CNN Encoder ---
        # Takes a (C, H, W) frame and outputs a feature vector.
        # Here we assume input patches are single-channel (T, H, W) which we process frame by frame.
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # e.g., 256x256 -> 128x128
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # e.g., 128x128 -> 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4), # e.g., 64x64 -> 16x16
            
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
            nn.Flatten()
        )
        
        # A linear layer to project the CNN output to the desired size for the LSTM
        self.cnn_to_lstm = nn.Linear(64, cnn_output_size)

        # --- 2. LSTM for Temporal Modeling ---
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True, # expects input shape (batch, seq_len, features)
            bidirectional=False # Causal: process frames in order
        )

        # --- 3. Attention Mechanism ---
        self.attention = Attention(lstm_hidden_size)

        # --- 4. Classifier ---
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor (a batch of patches).
                              Shape: (batch_size, seq_length, height, width)
                              
        Returns:
            torch.Tensor: The raw logits for each class.
                          Shape: (batch_size, num_classes)
            torch.Tensor: The attention weights.
                          Shape: (batch_size, seq_length)
        """
        batch_size, seq_length, H, W = x.shape
        
        # --- CNN processing ---
        # We process each frame through the CNN.
        # Reshape to (batch_size * seq_length, 1, H, W) to process all frames at once.
        x_reshaped = x.view(batch_size * seq_length, 1, H, W)
        cnn_features = self.cnn_encoder(x_reshaped)
        
        # Project features to the size expected by the LSTM
        cnn_features = self.cnn_to_lstm(cnn_features)
        
        # Reshape back to a sequence for the LSTM: (batch_size, seq_length, cnn_output_size)
        lstm_input = cnn_features.view(batch_size, seq_length, -1)
        
        # --- LSTM processing ---
        # lstm_out shape: (batch_size, seq_length, lstm_hidden_size)
        # hidden shape: (num_layers, batch_size, lstm_hidden_size)
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        # --- Attention processing ---
        # context shape: (batch_size, lstm_hidden_size)
        # attn_weights shape: (batch_size, seq_length)
        context, attn_weights = self.attention(lstm_out)
        
        # --- Classifier ---
        context_with_dropout = self.dropout(context)
        logits = self.classifier(context_with_dropout)
        
        return logits, attn_weights


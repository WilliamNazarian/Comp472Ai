#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.NLLLoss()  # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer
    
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    
    for epoch in range(epochs):
        
        # Training
        model.train()  # Put model in training mode
        one_epoch_total_loss = 0  # Total loss for the current epoch
        correct_predictions = 0  # Correct predictions counter
        total_predictions = 0  # Total predictions counter
        
        for images, labels in train_loader:
            if torch.cuda.is_available():  # Use GPU instead of CPU
                images = images.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()  # Clear old gradients
            outputs_for_images = model(images)  # Forward pass
            print(f"Type of labels: {type(labels)}, Type of images: {type(images)}")
            print(f"Shape of labels: {labels.shape}, Shape of images: {images.shape}")
            loss = criterion(outputs_for_images, labels)  # Calculate the loss
            loss.backward()  # Calculate the gradients using back propagation
            optimizer.step()  # Adjust the parameters based on the gradients
            
            one_epoch_total_loss += loss.item()  # Accumulate the loss
            
            # Calculate the number of correct predictions
            _, predicted_classes = torch.max(outputs_for_images, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Calculate the average loss and accuracy for the epoch
        one_epoch_average_loss = one_epoch_total_loss / total_predictions
        accuracy = (correct_predictions / total_predictions) * 100
        
        train_loss.append(one_epoch_average_loss)
        train_accuracy.append(accuracy)
        
        # Validation
        model.eval()  # Put the network into evaluation mode
        loss_on_validation_set = 0  # Total loss for the current epoch
        correct_predictions = 0  # Correct predictions counter
        total_predictions = 0  # Total predictions counter
        
        with torch.no_grad():
            for images, labels in val_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                
                outputs_for_images = model(images)  # Do the forward pass
                loss_on_validation_set += criterion(outputs_for_images, labels).item()  # Calculate the loss
                
                # Record the correct predictions for validation data
                _, predicted = torch.max(outputs_for_images, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        # Record the validation loss and accuracy
        valid_loss.append(loss_on_validation_set / total_predictions)
        valid_accuracy.append((correct_predictions / total_predictions) * 100.0)
        
        print('Epoch %d/%d, Tr Loss: %.4f, Tr Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f' %
              (epoch + 1, epochs, train_loss[-1], train_accuracy[-1], 
               valid_loss[-1], valid_accuracy[-1]))

    


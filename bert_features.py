from data_loader import load_data

train, val, test = load_data('a')
print(train)
print(val)
print(test)

### Parameters ###
max_seq_length = 40
batch_size = 32
subtask = 'a'
learning_rate = 0.001

### Data ###
train_dataloader, val_dataloader, test_dataloader = get_data_bert(int(max_seq_length), batch_size, subtask)

### Model ###
model = models.BertFeatures(dropout, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = F.cross_entropy

### Training ###
train_metrics, val_metrics = train_and_evaluate(num_epochs, model, optimizer, loss_fn, train_dataloader, val_dataloader, early_stopping_criteria, directory_checkpoints, use_bert, use_mongo)

for epoch in trange(num_epochs, desc=batch_size='Epoch'):
    train_results = train_model(model, optimizer, loss_fn, train_dataloader, device, True)

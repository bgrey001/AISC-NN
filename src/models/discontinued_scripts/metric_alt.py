#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:22:26 2022

@author: benedict

alternate metric calculations
"""

# =============================================================================
# method to build confusion matrix
# =============================================================================
def confusion_matrix(self):
    
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    test_correct = 0
    total_f1 = 0
    
    predicted, labels = [], []
    n_correct = 0
    
    acc = 0
    
    total_preds = 0
    total_labels = 0
    confmatrix = torch.zeros(6, 6).to(self.device)
    
    class_list = ['drifting_longlines', 'fixed_gear', 'pole_and_line', 'purse_seines', 'trawlers', 'trollers'] # just for visual reference
    test_generator = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.train_data.CNN_collate)
    confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
    with torch.no_grad():
        self.model.eval()
        for test_features, test_labels, lengths in test_generator:
            test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)
            test_output = self.model(test_features)
            preds = torch.argmax(test_output, dim=1)
            
            # accuracy precision and recall usnig torch metrics
            # micro records average over all labels (what we want)
            # macro records mean of class averages
            # weighted records weighted averages using their support
            
            accuracy = multiclass_accuracy(test_output, test_labels, num_classes=self.n_classes, average='weighted').to(self.device)
            precision = multiclass_precision(test_output, test_labels, num_classes=self.n_classes, average=None).to(self.device)
            recall = multiclass_recall(test_output, test_labels, num_classes=self.n_classes, average=None).to(self.device)
            f1 = multiclass_f1_score(test_output, test_labels, num_classes=self.n_classes, average=None).to(self.device)
            
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            n_correct += ((preds == test_labels).sum().item())/len(test_labels)
            
            confmatrix += multiclass_confusion_matrix(test_output, test_labels, num_classes=self.n_classes)
            
            # sklearn confmat variables
            predicted.append(preds.cpu().detach().numpy())
            labels.append(test_labels.cpu().detach().numpy())

            correct = (preds == test_labels).sum().item()
            test_correct += (preds == test_labels).sum().item()

            for label, pred in zip(test_labels.view(-1), preds):
                confusion_matrix[label.long(), pred.long()] += 1
                
                
    
    predicted = np.concatenate(predicted).ravel().tolist()
    labels = np.concatenate(labels).ravel().tolist()
    
    
    a_score = accuracy_score(labels, predicted)
    
    precision, recall, f1_score, support = precision_recall_fscore_support(labels, predicted)
    confmat = cf(labels, predicted)
    # self.class_accuracies = (confmat.diag()/confmat.sum(1)).cpu().numpy() * 100

    report = classification_report(labels, predicted)
    
    self.history['confusion_matrix'] = confmat
    self.history['report'] = report
    self.history['class_precisions'] = precision
    self.history['class_recalls'] = recall
    self.history['class_F1_scores'] = f1_score
    self.history['class_supports'] = support
    
    print(f'CONFUSION MATRIX: \n{confmat}')
    print(f'CONFUSION MATRIX alt: \n{confmatrix.cpu().detach().numpy().astype(int)}')
    
    print(f'ACCURACY: {a_score * 100}')
    print(f'Accuracy torch metric: {total_accuracy / len(test_generator) * 100}')
    print(f'Accuracy custom: {n_correct / len(test_generator) * 100}')

    print(f'PRECISION: {precision}')
    print(f'PRECISION alt: {total_precision/len(test_generator)}')
    print(f'RECALL: {recall}')
    print(f'RECALL alt: {total_recall/len(test_generator)}')
    print(f'F1-SCORE: {f1_score}')
    print(f'F1-SCORE alt: {total_f1/len(test_generator)}')
    print(f'SUPPORT: {support}')
    
    
    
    # print(f'CLASSIFICATION REPORT: \n{report}')


    # longline accuracy = TP + TN / all
    # 12364 + (80 + )

    
    # print(type(y_true))
    # print(y_pred)
            
    # print(confmat)
    # print(f'Accuracy per class alt: {self.class_accuracies}')
    # print(f'Precision per class: {total_precision / len(test_generator) * 100}')
    # print(f'Recall per class: {total_recall / len(test_generator) * 100}')
    # print(f'F1 score per class: {total_f1 / len(test_generator) * 100}')

    # print(confusion_matrix)
    
    # print(f'Test accuracy 1: {self.history["test_accuracy"][-1]}')
    # print(f'Test accuracy 1: {100 * (test_correct / (len(test_generator)) * self.batch_size)}')
    
    # print(acc/len(test_generator))
    
    # print(self.class_accuracies * 100)
    # print('CONFUSION MATRIX:\n', confusion_matrix)
    # print('TOTAL TRUE POSITIVES = ', confusion_matrix.diag().sum())
    # print('TOTAL RESULTS = ', confusion_matrix.sum(1))

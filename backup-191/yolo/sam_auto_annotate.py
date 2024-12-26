from ultralytics.data.annotator import auto_annotate

auto_annotate(data="test_imgs/counter_standing/HAE2240625045735CA146-3_4.jpg", det_model="models_and_labels/counter_standing.pt", sam_model="pretrained/sam_b.pt")
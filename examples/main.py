import threshold_image_maker

imaker = threshold_image_maker.ThresholdImageMaker(threshold=160)
imaker.save_binary_image('img.jpeg', 'ok.png', threshold_mode='adaptive', clean_image=True, transparent_background=True)
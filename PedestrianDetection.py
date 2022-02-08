
import cv2 as cv

video_path = "C:/MEDIA_PIPE/Video/Videoplayback.mp4"

capture = cv.VideoCapture(video_path)

hog_obj = cv.HOGDescriptor()
pple_detector = cv.HOGDescriptor_getDefaultPeopleDetector()

hog_obj.setSVMDetector(pple_detector)

confidence = 0.6
scale_factor = 0.75
while True:
    result, image = capture.read()

    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_resize = cv.resize(image, (0,0), None, scale_factor, scale_factor)
        regions,weights= hog_obj.detectMultiScale(image_resize,
                                                  winStride=(4,4),
                                                  padding=(8,8),
                                                  scale=1.01)
        for region, weight in zip(regions, weights):
            if weight > confidence:

                left, top, width, height = region


                cv.rectangle(image,
                             (left, top),
                             (left+width, top+height),
                             (30, 200, 30),
                             2)

        cv.imshow("HOG-SVM", image)
        cv.waitKey(1)

cv.destroyAllWindows()
capture.release()
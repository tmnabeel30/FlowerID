//
//  ContentView.swift
//  FlowerID
//
//  Created by Nabeel on 08/04/23.
//

import SwiftUI
import UIKit
import Photos
import CoreML
import Vision

class PixelBufferGenerator {
    static func pixelBuffer(forImage image: CIImage, context: CIContext) -> CVPixelBuffer? {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        if status != kCVReturnSuccess {
            return nil
        }

        context.render(image, to: pixelBuffer!)
        return pixelBuffer
    }
}


struct ContentView: View {
    @State private var image: UIImage?
    @State private var predictedObject: String?
    @State private var showImagePicker: Bool = false
    @State private var showCamera: Bool = false
    
    var body: some View {
        VStack {
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
            } else {
                Text("No image selected")
            }
            
            Button("Select Image") {
                self.showImagePicker = true
            }
            .sheet(isPresented: $showImagePicker) {
                ImagePickerView(sourceType: .photoLibrary, selectedImage: self.$image, predictedObject: self.$predictedObject)
            }
            
            Button("Take Photo") {
                self.showCamera = true
            }
            .sheet(isPresented: $showCamera) {
                ImagePickerView(sourceType: .camera, selectedImage: self.$image, predictedObject: self.$predictedObject)
            }
            Text("Predicted object: \(predictedObject ?? "Unknown")")

            
            if let predictedObject = predictedObject {
                Text("Predicted object: \(predictedObject)")
            }
        }
    }
}

struct ImagePickerView: UIViewControllerRepresentable {
    typealias UIViewControllerType = UIImagePickerController
    typealias Coordinator = ImagePickerCoordinator
    
    var sourceType: UIImagePickerController.SourceType
    @Binding var selectedImage: UIImage?
    @Binding var predictedObject: String?
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        picker.allowsEditing = false
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // No need to update anything here
    }
    
    func makeCoordinator() -> ImagePickerCoordinator {
        return ImagePickerCoordinator(selectedImage: $selectedImage, predictedObject: $predictedObject)
    }
}

class ImagePickerCoordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    @Binding var selectedImage: UIImage?
    @Binding var predictedObject: String?
    
    init(selectedImage: Binding<UIImage?>, predictedObject: Binding<String?>) {
        _selectedImage = selectedImage
        _predictedObject = predictedObject
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let image = info[.originalImage] as? UIImage {
            selectedImage = image
            detectObjectsInImage(image) { result in
                if let result = result {
                    let (classLabel, probability) = result
                    self.predictedObject = "\(classLabel) (\(String(format: "%.2f", probability * 100))%)"
                } else {
                    self.predictedObject = "Unknown"
                }
            }
        }
        
        picker.dismiss(animated: true)
    }
}
func detectObjectsInImage(_ image: UIImage, completion: @escaping ((classLabel: String, probability: Double)?) -> Void) {
    guard let ciImage = CIImage(image: image) else {
        completion(nil)
        return
    }

    let context = CIContext(options: nil)
    guard let pixelBuffer = PixelBufferGenerator.pixelBuffer(forImage: ciImage, context: context) else {
        completion(nil)
        return
    }
    
    do {
        let model = try Oxford102(configuration: MLModelConfiguration())
        let vnModel = try VNCoreMLModel(for: model.model)
        
        let request = VNCoreMLRequest(model: vnModel) { request, error in
            if let results = request.results as? [VNClassificationObservation], let firstResult = results.first {
                let classLabel = firstResult.identifier
                let probability = firstResult.confidence
                completion((classLabel: classLabel, probability: Double(probability)))
            } else {
                completion(nil)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try handler.perform([request])
    } catch {
        completion(nil)
    }
}

struct ContentView_Previews: PreviewProvider {
static var previews: some View {
ContentView()
}
}
 

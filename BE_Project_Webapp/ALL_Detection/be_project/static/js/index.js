const image = document.getElementById("imageInput");
const display_image = document.getElementById("display_image");

image.onchange = function handleFileUpload(event) {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (readerEvent) {
        const image = new Image();
        image.onload = function () {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            // Set the canvas dimensions to 256x256
            canvas.width = 256;
            canvas.height = 256;

            // Draw the original image onto the canvas
            ctx.drawImage(image, 0, 0, 256, 256);

            // Create a new image element to display the resized image
            const resizedImage = new Image();

            // Set the src of the new image to the canvas data URL
            display_image.src = canvas.toDataURL('image/jpeg');
        };

        image.src = readerEvent.target.result;
        };

        reader.readAsDataURL(file);
    } else {
        alert('Please upload an image file.');
    }
}


const label = document.querySelector(".output-label")
console.log(label.innerHTML)

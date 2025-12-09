// app/static/js/app.js

let videoStream = null;
let detectedItem = null;

async function startCamera() {
  const video = document.getElementById("video");
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = videoStream;
  } catch (err) {
    console.error("Error accessing camera:", err);
    alert("Could not access camera. Check permissions.");
  }
}

function captureFrameAndPredict() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw current video frame onto canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Get base64-encoded image
  const dataURL = canvas.toDataURL("image/jpeg");

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ image: dataURL })
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        console.error("Prediction error:", data.error);
        alert("Prediction failed: " + data.error);
      } else {
        detectedItem = data.item;
        const confidence = data.confidence;
        document.getElementById("detectedItem").textContent = detectedItem;
        document.getElementById("confidenceText").textContent =
          `Confidence: ${(confidence * 100).toFixed(2)}%`;
      }
    })
    .catch(err => {
      console.error("Fetch /predict error:", err);
      alert("Prediction request failed.");
    });
}

function addItem() {
  const quantityInput = document.getElementById("quantityInput");
  const message = document.getElementById("message");

  if (!detectedItem) {
    alert("Please detect an item first.");
    return;
  }

  const quantity = quantityInput.value.trim();
  if (!quantity) {
    alert("Please enter a quantity.");
    return;
  }

  fetch("/add_item", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ item: detectedItem, quantity: quantity })
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        alert("Error adding item: " + data.error);
      } else {
        message.textContent = "Item added successfully!";
        quantityInput.value = "";
        loadItems();
        setTimeout(() => { message.textContent = ""; }, 2000);
      }
    })
    .catch(err => {
      console.error("Fetch /add_item error:", err);
      alert("Failed to add item.");
    });
}

function loadItems() {
  fetch("/items")
    .then(res => res.json())
    .then(data => {
      const tbody = document.getElementById("itemsTbody");
      tbody.innerHTML = "";

      (data.items || []).forEach(row => {
        const tr = document.createElement("tr");
        const tdItem = document.createElement("td");
        const tdQty = document.createElement("td");
        const tdTs = document.createElement("td");

        tdItem.textContent = row.item;
        tdQty.textContent = row.quantity;
        tdTs.textContent = row.timestamp;

        tr.appendChild(tdItem);
        tr.appendChild(tdQty);
        tr.appendChild(tdTs);

        tbody.appendChild(tr);
      });
    })
    .catch(err => {
      console.error("Fetch /items error:", err);
    });
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("startCameraBtn").addEventListener("click", startCamera);
  document.getElementById("captureBtn").addEventListener("click", captureFrameAndPredict);
  document.getElementById("addItemBtn").addEventListener("click", addItem);

  // Load existing dashboard data on page load
  loadItems();
});

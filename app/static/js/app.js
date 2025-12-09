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

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL("image/jpeg");

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
    headers: { "Content-Type": "application/json" },
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

// ---------- Temperature handling ----------
function setTemperature() {
  const tempInput = document.getElementById("tempInput");
  const value = tempInput.value.trim();
  if (!value) {
    alert("Please enter a temperature.");
    return;
  }

  fetch("/set_temp", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ temperature: value })
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        alert("Error setting temperature: " + data.error);
      } else {
        document.getElementById("bigTempDisplay").textContent =
          `${data.setTemp} °C`;

        // Recompute shelf lives, so refresh items
        loadItems();
      }
    })
    .catch(err => {
      console.error("Fetch /set_temp error:", err);
      alert("Failed to set temperature.");
    });
}

function loadCurrentTemperature() {
  fetch("/get_temp")
    .then(res => res.json())
    .then(data => {
      const current = data.setTemp;
      const bigDisplay = document.getElementById("bigTempDisplay");
      const tempInput = document.getElementById("tempInput");

      if (current === null || current === undefined) {
        bigDisplay.textContent = "-- °C";
      } else {
        bigDisplay.textContent = current + " °C";
        tempInput.value = current;
      }
    })
    .catch(err => {
      console.error("Fetch /get_temp error:", err);
    });
}

// ---------- Load items (now including setTemp) ----------
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
        const tdShelf = document.createElement("td");
        const tdTs = document.createElement("td");

        tdItem.textContent = row.item;
        tdQty.textContent = row.quantity;
        tdShelf.textContent =
          row.shelfLife !== null && row.shelfLife !== undefined
            ? row.shelfLife + " s"
            : "(unknown)";
        tdTs.textContent = row.timestamp;

        tr.appendChild(tdItem);
        tr.appendChild(tdQty);
        tr.appendChild(tdShelf);
        tr.appendChild(tdTs);

        tbody.appendChild(tr);
      });
    })
    .catch(err => {
      console.error("Fetch /items error:", err);
    });
}

// ---------- Init ----------
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("startCameraBtn").addEventListener("click", startCamera);
  document.getElementById("captureBtn").addEventListener("click", captureFrameAndPredict);
  document.getElementById("addItemBtn").addEventListener("click", addItem);

  document.getElementById("setTempBtn").addEventListener("click", setTemperature);

  loadCurrentTemperature();
  loadItems();
});

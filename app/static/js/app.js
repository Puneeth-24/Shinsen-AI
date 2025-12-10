// app/static/js/app.js
let videoStream = null;
let detectedItem = null;
let isEditingUseQty = false; 

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

function useItem(itemId, maxQty, inputEl) {
  const val = inputEl.value.trim();
  if (!val) {
    alert("Please enter a quantity to mark as used.");
    return;
  }

  const used = parseInt(val, 10);
  if (isNaN(used) || used <= 0) {
    alert("Used quantity must be a positive integer.");
    return;
  }
  if (used > maxQty) {
    alert(`You only have ${maxQty} of this item.`);
    return;
  }

  fetch("/use_item", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: itemId, usedQuantity: used })
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        alert("Error updating item: " + data.error);
      } else {
        // Reload table so new quantity / deletions show up
        loadItems();
      }
    })
    .catch(err => {
      console.error("Fetch /use_item error:", err);
      alert("Failed to update item.");
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
        const tdUse = document.createElement("td");

        tdItem.textContent = row.item;
        tdQty.textContent = row.quantity;
        tdShelf.textContent =
          row.shelfLife !== null && row.shelfLife !== undefined
            ? row.shelfLife + " s"
            : "(unknown)";
        tdTs.textContent = row.timestamp;

        // ---- Use controls: input + button ----
        const input = document.createElement("input");
        input.type = "number";
        input.min = "1";
        input.max = row.quantity != null ? String(row.quantity) : "";
        input.className =
          "w-16 rounded border border-slate-300 px-1 py-0.5 text-xs mr-2";

        // 👇 Pause auto-refresh while user is typing
        input.addEventListener("focus", () => {
          isEditingUseQty = true;
        });
        input.addEventListener("blur", () => {
          // Give a tiny delay in case blur is triggered before click
          setTimeout(() => {
            isEditingUseQty = false;
          }, 100);
        });

        const btn = document.createElement("button");
        btn.textContent = "Used";
        btn.className =
          "rounded bg-rose-600 text-white text-xs px-2 py-1 hover:bg-rose-700";
        btn.addEventListener("click", () => {
          useItem(row.id, row.quantity, input);
        });

        tdUse.appendChild(input);
        tdUse.appendChild(btn);

        tr.appendChild(tdItem);
        tr.appendChild(tdQty);
        tr.appendChild(tdShelf);
        tr.appendChild(tdTs);
        tr.appendChild(tdUse);

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
setInterval(() => {
  if (!isEditingUseQty) {
    loadItems();
  }
}, 1000);
});

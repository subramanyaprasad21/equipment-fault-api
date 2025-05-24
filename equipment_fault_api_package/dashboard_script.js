function sendPrediction() {
  const inputData = {
    temperature: parseFloat(document.getElementById('temperature').value),
    pressure: parseFloat(document.getElementById('pressure').value),
    vibration: parseFloat(document.getElementById('vibration').value),
    humidity: parseFloat(document.getElementById('humidity').value),
    equipment: parseInt(document.getElementById('equipment').value),
    location: parseInt(document.getElementById('location').value),
  };

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(inputData)
  })
  .then(response => response.json())
  .then(data => {
    const resultDiv = document.getElementById('result');
    console.log("Received from API:", data); // 👈 Add this to debug

    resultDiv.style.display = 'block';
    resultDiv.style.backgroundColor = data.prediction === 1 ? '#ff4d4d' : '#28a745';
    resultDiv.style.color = 'white';

    resultDiv.innerHTML = `
      <strong>Status:</strong> ${data.prediction === 1 ? 'FAULTY' : 'NOT FAULTY'}<br>
      <strong>Confidence:</strong><br>
      Faulty: ${Math.round(data.confidence.faulty * 100)}%<br>
      Not Faulty: ${Math.round(data.confidence.not_faulty * 100)}%
    `;
  })
  .catch(error => {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.style.color = 'red';
    resultDiv.innerText = 'Error calling API: ' + error;
  });
}

document.addEventListener('DOMContentLoaded', function () {
    // Convert UTC times to local timezone
    function convertScanTimesToLocal() {
        const scanTimeElements = document.querySelectorAll('.scan-time');
        
        scanTimeElements.forEach(element => {
            const utcTime = element.getAttribute('data-utc-time');
            if (utcTime && utcTime !== '') {
                try {
                    const date = new Date(utcTime);
                    if (!isNaN(date.getTime())) {
                        // Format to local time: YYYY-MM-DD HH:MM
                        const localTime = date.toLocaleString(undefined, {
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            hour12: false
                        });
                        element.textContent = localTime;
                    }
                } catch (error) {
                    console.error('Error converting time:', error);
                }
            }
        });
    }

    // Convert times on page load
    convertScanTimesToLocal();

    // Handle Run Scan button
    const runScanBtn = document.getElementById('run-scan-btn');
    if (runScanBtn) {
        runScanBtn.addEventListener('click', function () {
            fetch('/run_scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert(data.message);
                    // Refresh the page to show the new scan
                    setTimeout(() => {
                        location.reload();
                    }, 2000);
                }
            })
            .catch(error => {
                console.error('Error running scan:', error);
                alert('Error running scan. Please try again.');
            });
        });
    }
});

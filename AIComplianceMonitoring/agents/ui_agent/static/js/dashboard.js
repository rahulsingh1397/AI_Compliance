document.addEventListener('DOMContentLoaded', function () {
    // Function to show notifications
    function showNotification(message, type = 'info') {
        const container = document.querySelector('.container-fluid');
        if (!container) return;

        const alertType = type === 'error' ? 'danger' : type;
        const notificationHtml = `
            <div class="alert alert-${alertType} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        // Remove any existing alerts before showing a new one
        const existingAlert = container.querySelector('.alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        container.insertAdjacentHTML('afterbegin', notificationHtml);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }

    // Function to fetch compliance data from the API
    async function fetchComplianceData() {
        try {
            // TEMPORARY: Using test endpoint that bypasses authentication
            const response = await fetch('/test_compliance_data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const responseData = await response.json();
            console.log('Received data from API:', responseData); // Debug log to check data format
            
            // Handle different response formats
            if (responseData.data && responseData.status === 'success') {
                console.log('Detected nested data structure, extracting data property');
                return responseData.data;
            } else if (responseData.key_metrics) {
                console.log('Detected flat data structure');
                return responseData;
            } else if (responseData.report_id) {
                console.log('Detected raw report structure.');
                return responseData;
            } else {
                console.error('Unexpected data format:', responseData);
                // Return mock data to prevent dashboard errors
                return {
                    is_mock_data: true,
                    mock_generated_at: '2025-06-25 15:30:00',
                    key_metrics: {
                        sensitive_files: 42,
                        total_scans: 150,
                        risk_level: 'Medium',
                        last_scan_date: '2025-06-25 15:30:00'
                    },
                    sensitive_data_types: {
                        'High Priority': 3,
                        'Medium Priority': 12,
                        'Low Priority': 27
                    },
                    compliance_status: {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92},
                    recent_alerts: [
                        {timestamp: '2025-06-25 15:30:00', message: 'Mock alert 1', severity: 'High'},
                        {timestamp: '2025-06-25 14:25:00', message: 'Mock alert 2', severity: 'Medium'}
                    ]
                };
            }
        } catch (error) {
            console.error("Could not fetch compliance data:", error);
            // Display an error message to the user on the dashboard
            const dashboard = document.querySelector('.container-fluid');
            if (dashboard) {
                const errorHtml = `<div class="alert alert-danger" role="alert">Error loading dashboard data. Please try again later.</div>`;
                dashboard.insertAdjacentHTML('afterbegin', errorHtml);
            }
            return null;
        }
    }

    // Function to update the key metrics
    function updateKeyMetrics(data) {
        if (!data) return;

        const safeUpdate = (id, value) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value || '-';
            }
        };

        try {
            const metrics = data.key_metrics || data;
            if (!metrics) return;

            safeUpdate('sensitive-files-count', metrics.sensitive_files);
            safeUpdate('total-scans-count', metrics.total_scans);

            let riskLevel = metrics.risk_level || '-';
            if (typeof riskLevel === 'number') {
                riskLevel = riskLevel === 3 ? 'High' : (riskLevel === 2 ? 'Medium' : 'Low');
            }
            safeUpdate('risk-level-status', riskLevel);

            const lastScanDate = metrics.last_scan_date;
            if (lastScanDate && lastScanDate !== 'N/A') {
                // Correctly parse UTC and display in user's local time
                const utcDate = new Date(lastScanDate.replace(' ', 'T') + 'Z');
                if (!isNaN(utcDate.getTime())) {
                    const localDateString = utcDate.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'medium' });
                    safeUpdate('last-scan-date', localDateString);
                } else {
                    safeUpdate('last-scan-date', 'Invalid Date');
                }
            } else {
                safeUpdate('last-scan-date', 'N/A');
            }

        } catch (error) {
            console.error('Error updating key metrics:', error);
        }
    }

    // Function to render the sensitive data types chart
    function renderDataTypesChart(data) {
        const canvas = document.getElementById('sensitive-data-types-chart');
        if (!canvas) return; // Don't run if the chart element isn't on the page
        if (!data) return;
        
        // Get sensitive data types, handling both nested and flat structures
        let dataTypes = {};
        if (data.sensitive_data_types) {
            dataTypes = data.sensitive_data_types;
        } else if (data.data_types) {
            dataTypes = data.data_types;
        } else {
            // Default data if none available
            dataTypes = {
                'Financial': 3,
                'Health': 1,
                'PII': 8
            };
            console.log('No sensitive data types found, using defaults');
        }
        
        const ctx = document.getElementById('sensitive-data-types-chart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(dataTypes),
                datasets: [{
                    data: Object.values(dataTypes),
                    backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
                    hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf'],
                    hoverBorderColor: "rgba(234, 236, 244, 1)",
                }],
            },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed !== null) {
                                    label += context.parsed + ' files';
                                }
                                return label;
                            }
                        }
                    }
                }
            },
        });
    }

    // Function to render the compliance status chart
    function renderComplianceStatusChart(data) {
        const canvas = document.getElementById('compliance-status-chart');
        if (!canvas) return; // Don't run if the chart element isn't on the page
        if (!data) return;
        
        // Get compliance status, handling both nested and flat structures
        let complianceData = {};
        if (data.compliance_status) {
            complianceData = data.compliance_status;
        } else {
            // Default data if none available
            complianceData = {
                'GDPR': 85,
                'CCPA': 70,
                'HIPAA': 90
            };
            console.log('No compliance status found, using defaults');
        }
        
        const ctx = document.getElementById('compliance-status-chart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(complianceData),
                datasets: [{
                    label: 'Compliance %',
                    data: Object.values(complianceData),
                    backgroundColor: '#4e73df',
                    borderColor: '#4e73df',
                    borderWidth: 1
                }]
            },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%'
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Function to populate the recent alerts table
    function populateAlertsTable(data) {
        const tableBody = document.getElementById('alerts-table-body');
        if (!tableBody) return; // Don't run if the table isn't on the page

        tableBody.innerHTML = ''; // Clear existing rows

        const alerts = data.recent_alerts || [];
        if (!Array.isArray(alerts) || alerts.length === 0) {
            console.log('No recent alerts to display.');
            const row = tableBody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 5;
            cell.textContent = 'No recent alerts found.';
            cell.style.textAlign = 'center';
            return;
        }

        alerts.forEach(alert => {
            const row = tableBody.insertRow();

            // Column 1: Description
            const descCell = row.insertCell(0);
            descCell.innerHTML = `
                <div class="alert-description">
                    <span class="fw-bold">${alert.message || 'No message'}</span>
                    <span class="text-muted d-block">${alert.location || 'Unknown Location'}</span>
                </div>`;

            // Column 2: Date
            const dateCell = row.insertCell(1);
            dateCell.textContent = alert.timestamp ? new Date(alert.timestamp).toLocaleString() : 'N/A';

            // Column 3: Severity
            const severityCell = row.insertCell(2);
            const severity = alert.severity || 'Unknown';
            severityCell.innerHTML = `<span class="severity-indicator severity-${severity.toLowerCase()}">${severity}</span>`;

            // Column 4: Status
            const statusCell = row.insertCell(3);
            const status = alert.status || 'New';
            statusCell.innerHTML = `<span class="status-indicator ${status.toLowerCase().replace(' ', '-')}">${status}</span>`;

            // Column 5: Actions
            const actionCell = row.insertCell(4);
            const alertId = alert.id || new Date(alert.timestamp).getTime();
            actionCell.innerHTML = `<a href="/dashboard/alert/${alertId}" class="btn-view-details"><i class="fas fa-chevron-right"></i></a>`;
        });
    }

    // Function to show or hide mock data indicator
    function showMockDataIndicator(isMockData, timestamp) {
        console.log('showMockDataIndicator called with:', isMockData, timestamp);
        
        let indicator = document.getElementById('mock-data-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'mock-data-indicator';
            indicator.style.position = 'fixed';
            indicator.style.top = '70px';
            indicator.style.right = '20px';
            indicator.style.backgroundColor = '#ff9800';
            indicator.style.color = 'white';
            indicator.style.padding = '8px 16px';
            indicator.style.borderRadius = '4px';
            indicator.style.fontWeight = 'bold';
            indicator.style.zIndex = '1000';
            indicator.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
            document.body.appendChild(indicator);
        }
        
        if (isMockData) {
            let timeDisplay = timestamp ? ` (${timestamp})` : '';
            indicator.textContent = `Demo Mode${timeDisplay}`;
            indicator.style.display = 'block';
        } else {
            indicator.style.display = 'none';
        }
    }

    // Function to handle scan button click
    function handleRunScan() {
        const runScanBtn = document.getElementById('run-scan-btn');
        if (!runScanBtn) return;
        
        runScanBtn.addEventListener('click', async function() {
            runScanBtn.disabled = true;
            runScanBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Starting scan...';
            
            try {
                const response = await fetch('/run_scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                
                const result = await response.json();
                showNotification(result.message, 'success');
                
                setTimeout(() => { initializeDashboard(); }, 5000);
                
            } catch (error) {
                console.error('Error running scan:', error);
                showNotification('Error starting scan. Please try again later.', 'error');
            } finally {
                runScanBtn.disabled = false;
                runScanBtn.innerHTML = '<i class="fas fa-search me-1"></i>Run Scan';
            }
        });
    }

    // Function to handle report creation button click
    function handleCreateReport() {
        const createReportBtn = document.getElementById('create-report-btn');
        if (!createReportBtn) return;

        createReportBtn.addEventListener('click', () => {
            showNotification('Creating report...', 'info');
            
            fetch('/reports/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showNotification(data.message, 'success');
                } else {
                    showNotification(`Error: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                console.error('Error creating report:', error);
                showNotification('An unexpected error occurred.', 'error');
            });
        });
    }

    // Main function to initialize the dashboard
    async function initializeDashboard() {
        const data = await fetchComplianceData();
        if (data) {
            const isMockData = data.is_mock_data === true;
            const mockTimestamp = data.mock_generated_at || null;
            
            showMockDataIndicator(isMockData, mockTimestamp);
            
            updateKeyMetrics(data);
            renderDataTypesChart(data);
            renderComplianceStatusChart(data);
            populateAlertsTable(data);
        }
    }

    // Initialize dashboard and set up event handlers
    initializeDashboard();
    handleRunScan();
    handleCreateReport();
});

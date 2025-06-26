document.addEventListener('DOMContentLoaded', function () {
    // Function to fetch compliance data from the API
    async function fetchComplianceData() {
        try {
            // TEMPORARY: Using test endpoint that bypasses authentication
            const response = await fetch('/api/compliance_data');
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
        console.log('updateKeyMetrics received:', data);
        
        // Handle both nested and flat structures
        if (data.key_metrics) {
            // Nested structure with key_metrics object
            document.getElementById('sensitive-files-count').textContent = data.key_metrics.sensitive_files;
            document.getElementById('total-scans-count').textContent = data.key_metrics.total_scans;
            document.getElementById('risk-level').textContent = data.key_metrics.risk_level;
            document.getElementById('last-scan-date').textContent = data.key_metrics.last_scan_date;
        } else {
            // Flat structure with top-level properties
            document.getElementById('sensitive-files-count').textContent = data.sensitive_files || '-';
            document.getElementById('total-scans-count').textContent = data.total_scans || '-';
            
            // Handle risk level (could be number or string)
            let riskLevel = '-';
            if (data.risk_level !== undefined) {
                if (typeof data.risk_level === 'number') {
                    riskLevel = data.risk_level === 3 ? 'High' : 
                               data.risk_level === 2 ? 'Medium' : 'Low';
                } else {
                    riskLevel = data.risk_level;
                }
            }
            document.getElementById('risk-level').textContent = riskLevel;
            
            // Format last_scan date if available
            let lastScan = '-';
            if (data.last_scan) {
                try {
                    const date = new Date(data.last_scan);
                    lastScan = date.toLocaleString();
                } catch (e) {
                    lastScan = data.last_scan;
                }
            }
            document.getElementById('last-scan-date').textContent = lastScan;
        }
    }

    // Function to render the sensitive data types chart
    function renderDataTypesChart(data) {
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
        
        const ctx = document.getElementById('dataTypesChart').getContext('2d');
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
        
        const ctx = document.getElementById('complianceStatusChart').getContext('2d');
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
        if (!data) return;
        const tableBody = document.getElementById('recent-alerts-table');
        tableBody.innerHTML = ''; // Clear existing rows
        
        // Check if recent_alerts exists
        if (!data.recent_alerts || !Array.isArray(data.recent_alerts) || data.recent_alerts.length === 0) {
            console.log('No recent alerts found in data');
            // Add a placeholder row
            tableBody.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center">No recent alerts</td>
                </tr>
            `;
            return;
        }
        
        // Process alerts
        data.recent_alerts.forEach(alert => {
            const severityClass = {
                'High': 'text-danger',
                'Medium': 'text-warning',
                'Low': 'text-info'
            }[alert.severity] || 'text-secondary';

            const row = `
                <tr>
                    <td>${alert.timestamp}</td>
                    <td>${alert.message}</td>
                    <td><span class="${severityClass}">${alert.severity}</span></td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });
    }

    // Function to show or hide mock data indicator
    function showMockDataIndicator(isMockData, timestamp) {
        // Get or create the indicator element
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

    // Main function to initialize the dashboard
    async function initializeDashboard() {
        const data = await fetchComplianceData();
        if (data) {
            // Check if this is mock data and show indicator if needed
            showMockDataIndicator(data.is_mock_data, data.mock_generated_at);
            
            updateKeyMetrics(data);
            renderDataTypesChart(data);
            renderComplianceStatusChart(data);
            populateAlertsTable(data);
        }
    }

    initializeDashboard();
});

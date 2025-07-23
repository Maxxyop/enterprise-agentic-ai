def interpret_output(zap_output):
    """
    Processes the output from ZAP and extracts relevant information.

    Args:
        zap_output (dict): The output from ZAP in dictionary format.

    Returns:
        dict: A dictionary containing extracted findings and relevant details.
    """
    findings = []
    
    for alert in zap_output.get('alerts', []):
        finding = {
            'url': alert.get('url'),
            'risk': alert.get('risk'),
            'description': alert.get('description'),
            'solution': alert.get('solution'),
            'evidence': alert.get('evidence'),
            'timestamp': alert.get('timestamp'),
        }
        findings.append(finding)

    return {
        'total_findings': len(findings),
        'findings': findings
    }
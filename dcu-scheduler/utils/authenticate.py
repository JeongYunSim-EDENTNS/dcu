from keycloak import KeycloakOpenID


def get_token(
    server_url: str,
    realm_name: str,
    client_id: str,
    client_secret_key: str,
    username: str,
    password: str
):
    keycloak_openid = KeycloakOpenID(
        server_url=server_url,
        realm_name=realm_name,
        client_id=client_id,
        client_secret_key=client_secret_key
    )

    token_data = keycloak_openid.token(username, password)
    token = token_data["access_token"]
    return token
